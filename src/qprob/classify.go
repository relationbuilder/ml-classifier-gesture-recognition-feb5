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
	Buckets   map[int16]*QuantList
	Spec      *CSVCol
	Enabled   bool
	ProcType  QuantType
	EffMaxVal float32
	EffMinVal float32
	BuckSize  float32
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
}

// A set of classifiers indexed by
// name to allow a single process
// to serve data from multiple training
// data sets
type Classifiers struct {
	classifiers map[string]*Classifier
}

/*func LoadClassifierFile(fiName string ) *Classifier {

}*/

/* separated out the makeEmptyClassifier so
accessible by other construction techniques
like from a string */
func makEmptyClassifier(fiName string, label string) *Classifier {
	fier := new(Classifier)
	fier.TrainFiName = fiName
	fier.Label = label
	fier.ClassCol = 0
	// can not set ColDef until we know how many
	// colums to allocate space for.
	//fier.ColDef = make([]*Feature, 0fier.Info.NumCol)
	fier.ColByName = make(map[string]*Feature)
	fier.ColNumByName = make(map[string]int)
	return fier
}

func makeFeature(col *CSVCol) *Feature {
	afeat := new(Feature)
	afeat.Spec = col
	afeat.Enabled = true
	afeat.BuckSize = 1
	afeat.EffMaxVal = col.MinFlt
	afeat.EffMinVal = col.MinFlt
	afeat.ProcType = QTBucket
	return afeat
}

func LoadClassifierTrainFile(fiName string, label string) *Classifier {
	fmt.Printf("fiName=%s", fiName)

	// Instantiate the basic File Information
	// NOTE: This early construction will have to
	// be duplciates for construction from
	// string when using in POST.
	fier := makEmptyClassifier(fiName, label)
	fier.Info = LoadCSVMetaDataFile("../data/breast-cancer-wisconsin.adj.data.csv")
	fier.NumCol = fier.Info.NumCol
	fier.ColDef = make([]*Feature, fier.NumCol)
	fier.Info.BuildDistMatrixFile()
	fmt.Println("loadCSVMetadata complete")
	fmt.Println(fier.Info.String())

	for i := 0; i < fier.NumCol; i++ {
		col := fier.Info.Col[i]
		fier.ColDef[i] = makeFeature(col)
	}

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
	for i := 0; i < fier.NumCol; i++ {
		if i == fier.ClassCol {
			continue // skip the class
		}
		classId := fier.classId(vals)
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
			buckId := col.buckId(f32)
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
}
