package qprob

import (
	"bufio"
	"fmt"
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

// containst the count for each class
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
	BuckSize       float32
	NumBuck        int16
	NumBuckOveride bool
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
	Class int16
	Count int32
	Prob  float32
}

// ResultRow is the total set of probabilities
// along with feeding counts for each Row
// Not keeping Result items as pointer because
// we want to be able to feed this direclty
// into JSON formatter.
type ResultForFeat struct {
	Crecs  map[int16]ResultItem
	TotCnt int32
}

// we want to know the best chosent
// result along with the computed results
// for each feature.   In
// some applications with many classes we
// may need to reduce this.
type ResultForRow struct {
	best     ResultItem
	features []ResultForFeat
}

// Output is a structure that shows us the count
//  for each class plust the prob for each class
//  plus the chosen out

func (fier *Classifier) classRow(drow []float32) *ResultForRow {
	tout := new(ResultForRow)
	featOut := make([]ResultForFeat, fier.NumCol)
	tout.features = featOut
	for fc := 0; fc < fier.NumCol; fc++ {
		dval := drow[fc]
		feat := fier.ColDef[fc]
		cs := feat.Spec
		fwrk := &featOut[fc]
		if fc == fier.ClassCol {
			continue // skip the class
		}
		if feat.Enabled == false || cs.CanParseFloat == false {
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
				fBuckWrk.Class = classId
				fBuckWrk.Count = classCnt
				classProb := fier.ClassProb[classId]
				baseProb := float32(classCnt) / float32(buck.totCnt)
				fBuckWrk.Prob = baseProb * classProb
				fwrk.TotCnt += classCnt
				fwrk.Crecs[classId] = *fBuckWrk
			} // for class
		} // if buck exist
	} // for feat
	return tout
} // func

/* printClassifierModel */
/* LoadClassifierTrainingModel */

/*func LoadClassifierFile(fiName string ) *Classifier {

}*/

// Change the number of buckets for every feature
// where that feature has not already been overridden
func (cl *Classifier) setNumBuckDefault(nb int16) {
	for a := 0; a < cl.NumCol; a++ {
		feat := cl.ColDef[a]
		if feat.NumBuckOveride == false {
			feat.NumBuck = nb
		}
	}
}

// Set number of buckets for a feature and record
// the fact it's number of buckets has been overridden
// so we don't change it if the default numBuck for
// the entire classifer is changed in the future.
func (fe *Feature) setNumBuck(nb int16) {
	fe.NumBuck = nb
	fe.NumBuckOveride = true
}

// Compute BucketId for current data value for this
// feature.
// TODO: Need to add actual implementation for this
//  which is based on observed range with outliers
//  removed.
func (fe *Feature) bucketId(fier *Classifier, dval float32) int16 {
	return fe.Spec.buckId(dval)
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
	return fier
}

func (cl *Classifier) makeFeature(col *CSVCol) *Feature {
	afeat := new(Feature)
	afeat.Spec = col
	afeat.Enabled = true
	afeat.BuckSize = 1
	afeat.EffMaxVal = col.MinFlt
	afeat.EffMinVal = col.MinFlt
	afeat.ProcType = QTBucket
	afeat.NumBuck = cl.NumBuck
	afeat.NumBuckOveride = false
	return afeat
}

func LoadClassifierTrainFile(fiName string, label string, numBuck int16) *Classifier {
	fmt.Printf("fiName=%s", fiName)

	// Instantiate the basic File Information
	// NOTE: This early construction will have to
	// be duplciates for construction from
	// string when using in POST.
	fier := makEmptyClassifier(fiName, label, numBuck)
	fier.Info = LoadCSVMetaDataFile("../data/breast-cancer-wisconsin.adj.data.csv")
	fier.NumCol = fier.Info.NumCol
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
	return fier

}

/* Retrive the class id for the row based on
the column that has been chosen to be used for
class Id */
func (fier *Classifier) classId(vals []string) int16 {
	ctx := vals[fier.ClassCol]
	id, err := strconv.ParseInt(ctx, 10, 16)
	if err != nil {
		fmt.Printf("classId() Encountered int parsing error val=%v err=%v", ctx, err)
		return -9999
	}
	return int16(id)
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
			fmt.Printf("LoadClassifierTrain() Encountered float parsing error when not expected row=%v col=%v val=%v",
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
				abuck.Counts[classId] = 0
			} else {
				abuck.Counts[classId] += 1
			}
			abuck.totCnt += 1
		}
		//fmt.Printf("i=%v ctxt=%s f32=%v maxStr=%s minStr=%s\n", i, ctxt, f32, col.MaxStr, col.MinStr)
	} // for columns
}

func (fier *Classifier) updateClassProb() {
	// Update Class Probability for a given
	// row being in any class
	for classId, classCnt := range fier.ClassCounts {
		fier.ClassProb[classId] = float32(classCnt) / float32(fier.NumTrainRow)
	}
}

func LoadClassifierTrain(fier *Classifier, scanner *bufio.Scanner) {
	rowCnt := 0
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
