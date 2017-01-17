package qprob

import (
	"bufio"
	"fmt"
	"math"
	"strconv"
	s "strings"
	//"io"
	//"io/ioutil"
	"log"
	//"net/http"
	"os"
	//"os/exec"
)

const (
	QTBucket = 1 // Quantize using bucket logic
	QTInt    = 2 // Quantize with simple int index
	QTText   = 3 // Quantize as tokenized text field
)

type QuantType uint8

// contains the count for each class
// identfied as having at least one row.
// eg for bucket 10 we have 4 classes.
// each id=1, count=10, id=5, count=4
// totalCnt is the sum of the Counts
// for all classes
type QuantList struct {
	Counts map[int16]int32
	totCnt int32
}

// includes the data and descriptive meta
// data for 1 feature of data or in CSV this
// would be 1 column of data.
type Feature struct {
	Buckets        map[int16]*QuantList
	Spec           *CSVCol
	Enabled        bool
	ProcType       QuantType
	EffMaxVal      float32
	EffMinVal      float32
	EffRange       float32
	BuckSize       float32
	NumBuck        int16
	NumBuckOveride bool
	FeatWeight     float32
}

// Includes the metadaa and classifer
// data for a given classifer
type Classifier struct {
	ColDef       []*Feature
	ColNumByName map[string]int
	ColByName    map[string]*Feature
	Info         *CSVInfo
	TrainFiName  string
	TestFiName   string
	Label        string
	NumCol       int
	NumRow       int
	ClassCol     int
	NumBuck      int16
	ClassCounts  map[int16]int32   // total Count of all Records of Class
	ClassProb    map[int16]float32 // Prob of any record being in class
	NumTrainRow  int32
}

// A set of classifiers indexed by
// name to allow a single process
// to serve data from multiple training
// data sets
type Classifiers struct {
	classifiers map[string]*Classifier
}

// For results look up each feature for the
// row then compute a bucket Id for that  value
// and want to know the count and count and
// probability each class relative to the
// total probability of an item being in that
// bucket.
type ResultItem struct {
	Prob float32
}

// ResultRow is the total set of probabilities
// along with feeding counts for each Row
// Not keeping Result items as pointer because
// we want to be able to feed this direclty
// into JSON formatter.
type ResultForFeat struct {
	Cls    map[int16]ResultItem
	TotCnt int32
}

// we want to know the best chosent
// result along with the computed results
// for each feature.   In
// some applications with many classes we
// may need to reduce this.
type ResultForRow struct {
	BestClass int16
	BestProb  float32
	Classes   map[int16]ResultItem
	Features  []ResultForFeat
}

/* Determines reasonable effective Min/Max value range for
the statistically significant values that represent the bulk
of the data.  Without this feature the quantize process can
compute poor bucket sizes that are negatively effected by a very
small amoun of data.  We do this by dividing all inbound data
counts between 1,000 buckets across the absolute min/max range
then to find the effective range we scan from the top and bottom
until the next bucket would exteed the numRemove specification
then return the value equivilant to the last bucket on each end
that did not exceed the numRemove requirement.  This allows us
to remove the influence for a given fraction of the data extreeme
value data on each end when computing effective range which
provides a better computation of bucket size based on the dominant
data.  This is not always desirable but in stock data it is very
common to have 1% to 2% with extreeme values which ends up causing
buckets that are too large forcing values in the main distribution
into a smaller number of buckets than intended. If either value
for bottom or top meet each other before reachign the number of
records then returns a error. */
func (fe *Feature) setFeatEffMinMax(fier *Classifier, numRemMin int, numRemMax int) (float32, float32) {
	minCnt := 0
	maxCnt := 0
	csv := fe.Spec
	dist := csv.distCounts
	stepSize := csv.distStepSize

	// Scan from Bottom removing large numbers
	// We look ahead 1 so if it would place us
	// over our targetCnt then we abort.
	bndx := -1
	effMinVal := csv.MinFlt
	for ; numRemMin > (minCnt+dist[bndx+1]) && bndx < 999; bndx++ {
		//fmt.Printf("L144: bndx=%v effMinVal=%v minCnt=%v nextDist=%v numRemMin=%v\n",
		//	bndx, effMinVal, minCnt, dist[bndx+1], numRemMin)
		minCnt += dist[bndx+1]
		effMinVal += stepSize
	}

	// Scan From Top removing large numbers
	// We look back by 1 so if that value would put us
	// over target removed then abort based on curr Val.
	tndx := 1000
	effMaxVal := csv.MaxFlt
	for ; numRemMax > (maxCnt+dist[tndx-1]) && tndx > 1; tndx-- {
		maxCnt += dist[tndx-1]
		//fmt.Printf("L155: tndx=%v effMaxVal=%v maxCnt=%v nextDist=%v numRemMax=%v\n",
		//	tndx, effMaxVal, maxCnt, dist[tndx-1], numRemMax)
		effMaxVal -= stepSize
	}
	if effMaxVal > effMinVal {
		fe.EffMinVal = effMinVal
		fe.EffMaxVal = effMaxVal
		fier.updateStepValues(fe)
	} else {
		//fmt.Printf("ERR effMax < effMin in setFeatEffMinMax")
	}
	return fe.EffMinVal, fe.EffMaxVal
}

// set Min Max Effective Range for every feature.
func (fier *Classifier) SetEffMinMax(numRemMin int, numRemMax int) {
	for cn := 0; cn < fier.NumCol; cn++ {
		fe := fier.ColDef[cn]
		_, _ = fe.setFeatEffMinMax(fier, numRemMin, numRemMax)

		csv := fe.Spec
		fmt.Printf("SetEffMinMax colNum=%v absMin=%v effMin=%v absMax=%v effMax=%v remMin=%v remMax=%v\n",
			cn, csv.MinFlt, fe.EffMinVal, csv.MaxFlt, fe.EffMaxVal, numRemMin, numRemMax)
	}
}

// Remove the top portion of the outlier records
// where portion of set is a ratio expressed between
// 0.0 and 1.0.  Applied to all features.
func (fier *Classifier) SetEffMinMaxPortSet(portSet float32) {
	numRemoveRec := int(float32(fier.NumRow) * portSet)
	fier.SetEffMinMax(numRemoveRec, numRemoveRec)
}

func (fier *Classifier) ClassRowStr(astr string) *ResultForRow {
	a := ParseStrAsArrFloat(astr)
	if len(a) < fier.NumCol {
		fmt.Println("classRowStr inputStr has wrong num fields numFld=%v numExpect=%v astr=%v",
			len(a), fier.NumCol, astr)
		return nil
	}
	return fier.ClassRow(a)
}

func (fier *Classifier) totFeatWeight() float32 {
	tout := float32(0.0)
	for ndx, feat := range fier.ColDef {
		if feat.Enabled == true && ndx != fier.ClassCol {
			tout += feat.FeatWeight
		}
	}
	return tout
}

// Output is a structure that shows us the count
//  for each class plust the prob for each class
//  plus the chosen out

func (fier *Classifier) ClassRow(drow []float32) *ResultForRow {
	tout := new(ResultForRow)
	tout.Classes = make(map[int16]ResultItem)
	clsm := make(map[int16]*ResultItem)
	featOut := make([]ResultForFeat, fier.NumCol)
	for fc := 0; fc < fier.NumCol; fc++ {
		featOut[fc].Cls = make(map[int16]ResultItem)
	}
	tout.Features = featOut
	for fc := 0; fc < fier.NumCol; fc++ {
		dval := drow[fc]
		feat := fier.ColDef[fc]
		//cs := feat.Spec
		fwrk := &featOut[fc]
		if fc == fier.ClassCol {
			continue // skip the class
		}
		if dval == math.MaxFloat32 {
			continue // skip if value contains invalid value
		}
		if feat.Enabled == false {
			continue
		}
		buckId := feat.bucketId(fier, dval)
		buck, bfound := feat.Buckets[buckId]

		if bfound == true {
			// Our training data set includes
			// at least row that had one feature
			// that contained one value the derived
			// to the same bucket Id.

			for classId, classCnt := range buck.Counts {
				// Ieterate over the classes that match
				// and record them for latter use.
				fBuckWrk := new(ResultItem)
				//classProb := fier.ClassProb[classId]
				baseProb := float32(classCnt) / float32(buck.totCnt)
				fBuckWrk.Prob = baseProb
				fwrk.TotCnt += classCnt
				fwrk.Cls[classId] = *fBuckWrk
				clswrk, clsFound := clsm[classId]
				if clsFound == false {
					clswrk = new(ResultItem)
					clsm[classId] = clswrk
				}
				clswrk.Prob += baseProb * feat.FeatWeight
				//fmt.Printf("col%v val=%v buck=%v class=%v baseProb=%v outProb=%v\n",
				//	fc, dval, buckId, classId, baseProb, fBuckWrk.Prob)
			} // for class
		} // if buck exist
	} // for feat

	// Copy Classes to acutal output
	// and select best item
	bestProb := float32(0.0)
	for classId, classWrk := range clsm {
		classWrk.Prob = classWrk.Prob / float32(fier.totFeatWeight())
		tout.Classes[classId] = *classWrk
		if bestProb < classWrk.Prob {
			bestProb = classWrk.Prob
			tout.BestProb = classWrk.Prob
			tout.BestClass = classId
		}
	}

	return tout
} // func

/* printClassifierModel */
/* LoadClassifierTrainingModel */

/* For an existing pre-loaded training data sett
add additional traiing data. This will not adjust
the bucket size or effective min, max values used
to compute step size for bucket indexing.   */
/*func AddClassifierTrainingFile(fiName string ) *Classifier {

}*/

// Change the number of buckets for every feature
// where that feature has not already been overridden
func (cl *Classifier) SetNumBuckDefault(nb int16) {
	for a := 0; a < cl.NumCol; a++ {
		feat := cl.ColDef[a]
		if feat.NumBuckOveride == false {
			feat.NumBuck = nb
			cl.updateStepValues(feat)
		}
	}
}

// Set number of buckets for a feature and record
// the fact it's number of buckets has been overridden
// so we don't change it if the default numBuck for
// the entire classifer is changed in the future.
func (cl *Classifier) SetNumBuck(fe *Feature, nb int16) {
	fe.NumBuck = nb
	fe.NumBuckOveride = true
	cl.updateStepValues(fe)
}

func (cl *Classifier) updateStepValues(afeat *Feature) {
	afeat.EffRange = afeat.EffMaxVal - afeat.EffMinVal
	afeat.BuckSize = afeat.EffRange / float32(afeat.NumBuck)
}

// Compute BucketId for current data value for this
// feature.
func (fe *Feature) bucketId(fier *Classifier, dval float32) int16 {
	// TODO: The simple multiplier process works when
	//  the numeric values are gauranteed between 0.0 and 1.0
	//  or are simple integer values.  A better
	//  process  computes a step size based on the value
	//  above minimum value divided by step size. The step size
	//  must be computed based on an effective range with
	//  with outliers removed
	amtOverMin := dval - fe.Spec.MinFlt
	bucket := int16(amtOverMin / float32(fe.BuckSize))
	//fmt.Printf("dval=%v bucket=%v amtOverMin=%v effMinVal=%v\n",
	//	dval, bucket, amtOverMin, fe.EffMinVal)
	return bucket
}

/* separated out the makeEmptyClassifier so
accessible by other construction techniques
like from a string */
func makEmptyClassifier(fiName string, label string, numBuck int16) *Classifier {
	fier := new(Classifier)
	fier.TrainFiName = fiName
	fier.Label = label
	fier.ClassCol = 0
	fier.NumTrainRow = 0
	fier.NumBuck = numBuck
	// can not set ColDef until we know how many
	// colums to allocate space for.
	//fier.ColDef = make([]*Feature, 0fier.Info.NumCol)
	fier.ColByName = make(map[string]*Feature)
	fier.ColNumByName = make(map[string]int)
	fier.ClassCounts = make(map[int16]int32)
	fier.ClassProb = make(map[int16]float32)
	return fier
}

func (fier *Classifier) initFromCSV(csv *CSVInfo) {
	fier.Info = csv
	fier.NumCol = fier.Info.NumCol
	fier.NumRow = fier.Info.NumRow
	fier.ColDef = make([]*Feature, fier.NumCol)
	fier.Info.BuildDistMatrixFile()
	fmt.Println("loadCSVMetadata complete")
	fmt.Println(fier.Info.String())

	// build up the feature description
	// for each column.
	for i := 0; i < fier.NumCol; i++ {
		col := fier.Info.Col[i]
		fier.ColDef[i] = fier.makeFeature(col)
	}
}

func (cl *Classifier) makeFeature(col *CSVCol) *Feature {
	afeat := new(Feature)
	afeat.Spec = col
	afeat.Enabled = true
	afeat.BuckSize = 1
	afeat.EffMaxVal = col.MaxFlt
	afeat.EffMinVal = col.MinFlt
	afeat.ProcType = QTBucket
	afeat.NumBuck = cl.NumBuck
	afeat.NumBuckOveride = false
	afeat.Buckets = make(map[int16]*QuantList)
	afeat.FeatWeight = 1.0
	cl.updateStepValues(afeat)
	return afeat
}

/* Retrive the class id for the row based on
the column that has been chosen to be used for
class Id */
func (fier *Classifier) classId(vals []string) int16 {
	ctx := vals[fier.ClassCol]
	id, err := strconv.ParseInt(ctx, 10, 16)
	if err != nil {
		//fmt.Printf("classId() Encountered int parsing error val=%v err=%v", ctx, err)
		return -9999
	}
	return int16(id)
}

// Update Class Probability for a given
// row being in any class This is used in
// later probability computations to adjust
// observed probability
func (fier *Classifier) updateClassProb() {

	for classId, classCnt := range fier.ClassCounts {
		fier.ClassProb[classId] = float32(classCnt) / float32(fier.NumTrainRow)
	}
}

/* For each value we retrive it's feature for
that column.  Figure out which bucket the current
value qualifies for and then update the count for
that class within that bucket. Creats the bucket and
the class if it is the first time we have seen them */
func (fier *Classifier) TrainRow(row int, vals []string) {
	fier.NumTrainRow += 1
	classId := fier.classId(vals)
	// Update Counts of all rows of
	// class so we can compute probability
	// of any row being of a given class
	_, classFnd := fier.ClassCounts[classId]
	if classFnd == false {
		fier.ClassCounts[classId] = 1
	} else {
		fier.ClassCounts[classId] += 1
	}

	for i := 0; i < fier.NumCol; i++ {
		if i == fier.ClassCol {
			continue // skip the class
		}

		col := fier.Info.Col[i]
		feat := fier.ColDef[i]

		if feat.Enabled == false {
			continue // no need testing disabled features
		}

		buckets := feat.Buckets
		ctxt := vals[i]
		if col.CanParseFloat == false {
			continue
		}

		// Record Min, Max Float
		f64, err := strconv.ParseFloat(ctxt, 32)
		if err != nil {
			fmt.Printf("LoadClassifierTrain() Encountered float parsing error when not expected row=%v col=%v val=%v\n",
				row, i, ctxt)
			col.CanParseFloat = false
		} else {
			f32 := float32(f64)
			// TODO: BuckId should be based on adjusted
			// range after outliers removed Just using this
			// for now to test the  rest.
			// TODO: If BuckId outside adjusted range
			// then may need to set to maxBuck, minBuck-1
			buckId := feat.bucketId(fier, f32)
			abuck, bexist := buckets[buckId]
			if bexist == false {
				abuck = new(QuantList)
				abuck.Counts = make(map[int16]int32)
				abuck.totCnt = 0
				buckets[buckId] = abuck
			}
			_, cntExist := abuck.Counts[classId]
			if cntExist == false {
				abuck.Counts[classId] = 1
			} else {
				abuck.Counts[classId] += 1
			}
			abuck.totCnt += 1
		}
		//fmt.Printf("i=%v ctxt=%s f32=%v maxStr=%s minStr=%s\n", i, ctxt, f32, col.MaxStr, col.MinStr)
	} // for columns
}

func LoadClassifierTrain(fier *Classifier, scanner *bufio.Scanner) {
	rowCnt := 0
	scanner.Scan() // skip headers
	for scanner.Scan() {
		txt := s.TrimSpace(scanner.Text())
		if err := scanner.Err(); err != nil {
			log.Fatal(err)
		}
		a := s.Split(txt, ",")
		if len(a) < fier.NumCol {
			continue
		}
		fier.TrainRow(rowCnt, a)
		rowCnt += 1
	} // for row
	fier.updateClassProb()
}

func LoadClassifierTrainFile(fiName string, label string, numBuck int16) *Classifier {
	fmt.Printf("fiName=%s", fiName)

	// Instantiate the basic File Information
	// NOTE: This early construction will have to
	// be duplciates for construction from
	// string when using in POST.
	fier := makEmptyClassifier(fiName, label, numBuck)
	csv := LoadCSVMetaDataFile(fiName)
	fier.initFromCSV(csv)

	// Now process the actual CSV file
	// to do the training work
	file, err := os.Open(fiName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Return the Header and use it's
	// contents to create the columns
	scanner := bufio.NewScanner(file)
	LoadClassifierTrain(fier, scanner)
	fier.SetEffMinMaxPortSet(0.015) // remove 1.5% of outlier records from top and bottom
	// when computing range.
	return fier

}
