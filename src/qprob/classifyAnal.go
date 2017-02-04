package qprob

// classifyAnal.go

import (
	"fmt"
)

const AnalNoClassSpecified = -9999

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

//If targClass is != AnalNoClassSpecified then the results
// for that class will be used as the driving input where
// as long as precision is >= targPrecision then increasing
// recall is chosen otherwise increasing precision is chosen.
// when targClass == AnalNoClassSpecified then increasing precision for entire
// data set is chosen.   This is used to pre-set the maxNumBuck
// for each column. In some instances it could be used to set
// MinNumBuck as well becaause if we know that a low number such
// as 2 yeidls poor results for a given class we do not want to
// allow the results module to fall back to those lower numbers
func (fier *Classifier) TestColumnNumBuck(targClass int16, targPrecis float32, trainRow [][]float32, testRow [][]float32) []*AnalResByClassByCol {
	return nil
}

// Runs each feature independantly by the number of buckets
// seeking the number of columns for this feature that return
// the best results.
func (fier *Classifier) TestIndividualColumnsNB(targClass int16, targPrecis float32, trainRows [][]float32, testRows [][]float32) []*AnalResByClass {
	// Question how do you quantify better.  If Precision is high
	// but recall is very low then which is better.  Seems like you
	// must set one as a minimum value and alllow the others to
	// vary.
	req := fier.Req
	featSet := make([]*Feature, 1)
	specClass := req.AnalClassId
	if specClass != AnalNoClassSpecified {
		fmt.Printf("Analyze for ClassId=%v\n", specClass)
	} else {
		fmt.Printf("Analyze for Total Set\n")
	}

	for _, feat := range fier.ColDef {
		featNum := feat.ColNum
		if featNum == fier.ClassCol {
			continue
		}
		startMaxNumBuck := feat.MaxNumBuck
		startMinNumBuck := feat.MinNumBuck
		featSet[0] = feat
		//featSet = fier.ColDef // see if changing individual feature changes entire set score

		_, sumRows := fier.ClassifyRows(testRows, featSet)
		startPrec := sumRows.Precis
		startRecall := float32(0.0)
		//fmt.Printf("L102: featNum=%v StartPrecis=%v startMaxNB=%v startMinNB=%v\n", featNum, startPrec, startMaxNumBuck, startMinNumBuck)
		bestMaxPrecis := sumRows.Precis
		bestMaxBuck := startMaxNumBuck
		bestMaxRecall := startRecall
		//fmt.Printf("specClass=%v AnalNoClassSpecified=%v\n", specClass, AnalNoClassSpecified)
		if specClass != AnalNoClassSpecified {
			clasSum := fier.MakeByClassStats(sumRows, testRows)
			tclass := clasSum.ByClass[specClass]
			//fmt.Printf("L113: Init by class tclass=%v\n", tclass)
			startRecall = tclass.Recall
			bestMaxRecall = tclass.Recall
			bestMaxPrecis = tclass.Prec
		}

		for maxNumBuck := feat.MaxNumBuck; maxNumBuck >= 2; maxNumBuck-- {
			feat.MaxNumBuck = maxNumBuck
			_, sumRows := fier.ClassifyRows(testRows, featSet)
			//fmt.Printf("L115: fe#=%v maxNB=%v setPrec=%v bMaxPrec=%v bMaxRec=%v bestNb=%v\n", featNum, maxNumBuck, sumRows.Precis, bestMaxPrecis, bestMaxRecall, bestMaxBuck)
			if req.AnalClassId == AnalNoClassSpecified {
				if sumRows.Precis >= bestMaxPrecis {
					// measure by accuracy when all rows are forced
					// to be classified eg: recall is forced to 100%
					// for the set by forcing the classifier to take
					// it's best guess for every row.
					bestMaxBuck = maxNumBuck
					bestMaxPrecis = sumRows.Precis
				}
			} else {
				// measure by target class or by set

				clasSum := fier.MakeByClassStats(sumRows, testRows)
				tclass := clasSum.ByClass[specClass]
				//fmt.Printf("L137: test by class tclass=%v\n", tclass)
				if (tclass.Prec > bestMaxPrecis && tclass.Recall >= bestMaxRecall) || (tclass.Prec >= bestMaxPrecis && tclass.Recall > bestMaxRecall) {
					bestMaxRecall = tclass.Recall
					bestMaxPrecis = tclass.Prec
					bestMaxBuck = maxNumBuck
				}
			}

		} // for maxNumBuck
		feat.MaxNumBuck = bestMaxBuck

		//fmt.Printf("L133: BEST MAX featNdx=%v  numBuck=%v Precis=%v\n", featNum, bestMaxBuck, bestMaxPrecis)
		//_, sumRows = fier.ClassifyRows(testRows, featSet)
		//fmt.Printf("L135: Retest with max  prec=%v\n", sumRows.Precis)

		// Now relax the minimum number of buckets to find our best setting
		bestMinBuck := startMinNumBuck
		bestMinPrecis := startPrec
		bestMinRecall := startRecall

		for minNumBuck := startMinNumBuck; minNumBuck <= feat.MaxNumBuck; minNumBuck++ {
			feat.MinNumBuck = minNumBuck
			_, sumRows := fier.ClassifyRows(testRows, featSet)

			//fmt.Printf("L145: fe#=%v minNB=%v SPrec=%v Bprec=%v BRecal=%v bNB=%v\n", featNum, minNumBuck, sumRows.Precis, bestMinPrecis, bestMinRecall, bestMinBuck)
			if req.AnalClassId == AnalNoClassSpecified {
				if sumRows.Precis >= bestMinPrecis {
					// measure by accuracy when all rows are forced
					// to be classified eg: recall is forced to 100%
					// for the set by forcing the classifier to take
					// it's best guess for every row.
					bestMinBuck = minNumBuck
					bestMinPrecis = sumRows.Precis
				}
			} else {
				// measure by target class or by set
				clasSum := fier.MakeByClassStats(sumRows, testRows)
				tclass := clasSum.ByClass[specClass]
				//fmt.Printf("L137: test by class tclass=%v\n", tclass)
				if (tclass.Prec > bestMinPrecis && tclass.Recall >= bestMinRecall) || (tclass.Prec >= bestMinPrecis && tclass.Recall > bestMinRecall) {
					bestMinRecall = tclass.Recall
					bestMinPrecis = tclass.Prec
					bestMinBuck = minNumBuck
				}
			}

		} // for minNumBuck
		feat.MinNumBuck = bestMinBuck
		//_, sumRows = fier.ClassifyRows(testRows, featSet)
		//fmt.Printf("L163:MIN fe#=%v BMinNB=%v BPrec=%v retestPre%v\n", featNum, bestMinBuck, bestMinPrecis, sumRows.Precis)

		// TODO: Add complete printout of what we discovered by Feature
		fmt.Printf("L158: After Analyze ColNum=%v colName=%v\n   startPrecis=%v endPrecis=%v\n  startRecall=%v endRecall=%v\n  startMaxNumBuck=%v endBackNumBuck=%v\n  startMinNumBuck=%v  endMinNumBuck=%v\n",
			feat.ColNum, feat.Spec.ColName, startPrec, bestMinPrecis, startRecall, bestMinRecall,
			startMaxNumBuck, bestMaxBuck, startMinNumBuck, bestMinBuck)
	} // for features
	_, sumRows := fier.ClassifyRows(testRows, fier.ColDef)
	fmt.Printf("L176: After analyze setPrec all Feat enabled = %v\n", sumRows.Precis)

	return nil
}
