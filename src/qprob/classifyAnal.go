package qprob

// classifyAnal.go

import (
	"fmt"
)

// Structures to report on results
// and make them easy to analyze
// Some of these are also used
// by the optimizer.   Not to be confused
// with ResultForRow which contains
// more detail such as probability for
// membership in each class.
type testRowRes struct {
	rowNum    int32
	actClass  int16
	predClass int16
}

type testRes struct {
	rows        []testRowRes
	cntRow      int32
	cntCorrect  int32
	percCorrect float32
}

func TestClassifyAnal() {
	fmt.Println("Hello World!")
}

/*
// Function Create Summary Results
func (fier *Classifier) createSummaryResults(astr string) *summaryResult {
	// NOTE: Some of this code already exists in ClassifyFiles
	return nil
}

func (sumRes *summaryResult) ToSimpleRowCSV(fier *Classifier) string {
	return ""

}


// function to build statistics by class
// from a given result set.


*/

type AnalResByClass struct {
	FeatNdx     float32
	minNumBuck  int16
	maxNumBuck  int16
	bestNumBuck int16
	ByClasses   map[int16]*ResByClass
}

type AnalResByClassByCol struct {
	Cols   []*AnalResByClass
	TotCnt int32
	SucCnt int32
	Prec   float32
	Recall float32
}

/*
//If targClass is != -9999 then the results
// for that class will be used as the driving input where
// as long as precision is >= targPrecision then increasing
// recall is chosen otherwise increasing precision is chosen.
// when targClass == 999 then increasing precision for entire
// data set is chosen.   This is used to pre-set the maxNumBuck
// for each column. In some instances it could be used to set
// MinNumBuck as well becaause if we know that a low number such
// as 2 yeidls poor results for a given class we do not want to
// allow the results module to fall back to those lower numbers
func (fier *Classifier, targClass int16, targPrecis float32, trainRow [][]float32, testRow [][]float32) TestColumnNumBuck() []*AnalResNumBuck {
	return nil
}

// Runs each feature independantly by the number of buckets
// seeking the number of columns for this feature that return
// the best results.
func (fier *Classifier, targClass int16, targPrecis float32, trainRow [][]float32, testRow [][]float32) TestIndividualColumnsNB() []*AnalByClass {
	// Question how do you quantify better.  If Precision is high
	// but recall is very low then which is better.  Seems like you
	// must set one as a minimum value and alllow the others to
	// vary.
	return nil
}
*/
