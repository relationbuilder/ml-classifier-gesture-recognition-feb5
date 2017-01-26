// classifyOptimizer.go
package qprob

import (
	"fmt"
	rand "math/rand"
	"qutil"
)

type OptFeature struct {
	numQuant int16
	priority float32
}

type OptSettings struct {
	features []OptFeature // settings for each features
}

type OptResByClass struct {
}

type OptResult struct {
}

func testClassifyOpt() {
	fmt.Println("Hello World!")
}

// Produce a semi random list of optimizer features
// we can ieterate across when choosing features to
// change.  It is only semi-random because we do want
// to force every feature to tested.
func (fier *Classifier) MakeOptFeatList(targLen int) []int16 {
	rand.Seed(int64(Nowms()))
	tout := make([]int16, targLen)
	numCol := len(fier.ColDef)
	maxCol := numCol - 1
	for len(tout) < targLen {
		for off := 0; off < numCol; off++ {
			maxOff := maxCol - off
			rr := int16(rand.Int31n(int32(off)))
			tout = append(tout, int16(rr))
			rr = int16(rand.Int31n(int32(off)))
			tout = append(tout, int16(rr))
			tout = append(tout, int16(off))
			rr = int16(rand.Int31n(int32(maxOff)))
			tout = append(tout, int16(rr))
			tout = append(tout, int16(maxOff))
		}
	}
	return tout
}

// run one optimizer pass return new precision and if we kept it.
func (fier *Classifier) optRunOne(featNdx int16, newNumBuck int16, newWeight float32, lastPrec float32, trainRows [][]float32, testRows [][]float32) (float32, bool) {
	feat := fier.ColDef[featNdx]
	oldWeight := feat.FeatWeight
	oldNumBuck := feat.NumBuck
	feat.FeatWeight = newWeight
	feat.NumBuck = newNumBuck
	currPrec := lastPrec
	if oldNumBuck != newNumBuck {
		fier.RetrainFeature(featNdx, trainRows)
	}
	_, sumRows := fier.ClassifyRows(testRows)
	fmt.Printf("lastPrec=%v  newPrec=%v", lastPrec, sumRows.Precis)
	keepFlg := false
	// This check is a little complex but if the precision improved
	// we definitely want to keep the change. But we still want to
	// keep the change as long as the precision didn't drop as long
	// as the complexity measured in numBuck or Weight dropped.
	if sumRows.Precis > lastPrec || (sumRows.Precis >= lastPrec && (newNumBuck < oldNumBuck || newWeight < oldWeight)) {
		keepFlg = true
		currPrec = sumRows.Precis
		fmt.Printf(" keep newNumBuck=%v newWeight=%v\n", newNumBuck, newWeight)
	} else {
		fmt.Printf(" revers newNumBuck=%v newWeight=%v\n", newNumBuck, newWeight)
		feat.NumBuck = oldNumBuck
		feat.FeatWeight = oldWeight
		currPrec = lastPrec
		if oldNumBuck != newNumBuck {
			fier.RetrainFeature(featNdx, trainRows)
		}
	} // reverse
	return currPrec, keepFlg
}

func (fier *Classifier) optRunOneFeat(featNdx int16, lastPrec float32, trainRows [][]float32, testRows [][]float32) (int16, float32) {
	// We can depend on optRunOne to reset the feature to
	// it's original value if there was no improvement.
	feat := fier.ColDef[featNdx]
	currPrec := lastPrec
	kept := false
	keepCnt := int16(0)

	// Try set weight to random number smaller than current weight
	if feat.FeatWeight > 0.0 {
		adjWeight := feat.FeatWeight * rand.Float32()
		currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, adjWeight, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}
	} else {
		currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, 1.0, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}
	}

	// Try set Weight to random number between 0 and maxWeight
	adjWeight := (100.0 * rand.Float32())
	currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, adjWeight, currPrec, trainRows, testRows)
	if kept {
		keepCnt++
	}

	// Try Num Buck to 2
	if feat.FeatWeight > 0.0 {
		currPrec, kept = fier.optRunOne(featNdx, 2, feat.FeatWeight, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}
	}

	// Try Set Num Buck to 1/2 current number
	if (feat.FeatWeight > 0.0) && (feat.NumBuck > 6) {
		adjNumBuck := feat.NumBuck / 2
		currPrec, kept = fier.optRunOne(featNdx, adjNumBuck, feat.FeatWeight, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}

	}

	// Try set NumBuck to a random value
	if feat.FeatWeight != 0 {
		adjNumBuck := int16(rand.Int31n(int32(500)))
		if adjNumBuck > 2 {
			currPrec, kept = fier.optRunOne(featNdx, adjNumBuck, feat.FeatWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}
		}
	}

	// Try first set weight and adjNum buck to random values
	if feat.FeatWeight != 0 {
		adjNumBuck := int16(rand.Int31n(int32(255)))
		adjWeight = (50.0 * rand.Float32())
		if adjNumBuck > 2 {
			currPrec, kept = fier.optRunOne(featNdx, adjNumBuck, adjWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}

		}
	}

	// Try set Weight to to 0
	currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, 0, currPrec, trainRows, testRows)
	if kept {
		keepCnt++
	}

	return keepCnt, currPrec
}

/* Each optimizer run must only use data from the test data set.
which means we must separate the test data into two sets so it
can be used as both training & test. To help combat over training
we stagger the data we select for the test / training split

As a simplifying assumption I will always test the entire test data
set forcing precision when the system is forced to make a gues at
every item.

  TODO:  An alternate strategy is to force the system to guess at
   only a subset of classes seeking a specified minimum of recall
   and specified minimum of recall for each of the subset of classes.


As a basic process we change either numBuck or feature weight for
a feature then re-runn the classify.  If it improves things based
on the specified rules then we keep the change.   and save what
we learned.

We always try to make changes first that would reduce effective
complexity which is Try to Turn features off,  Try to reduce
number of Buckets and try to keep priority close as possible to the
default priority of 1. The least complex system would have one
feature enabled with 2 buckets */
func (fier *Classifier) OptProcess(splitOneEvery int, maxTimeSec float64, targetPrecis float32) {
	fmt.Println("\nOptProcess label=%s TrainFi=%s")
	startTime := Nowms()
	//classes := fier.ClassIds()
	origTrainRows := fier.GetTrainRowsAsArr(OneGig)
	splitSkipPrefix := 1

	trainRows, testRows := qutil.SplitFloatArrOneEvery(origTrainRows, splitSkipPrefix, splitOneEvery)
	fier.Retrain(trainRows)
	//keepRunning := true
	featLst := fier.MakeOptFeatList(500)
	_, lastSum := fier.ClassifyRows(testRows)
	lastPrec := lastSum.Precis
	classCol := int16(fier.ClassCol)

	for _, featNdx := range featLst {
		if featNdx == classCol {
			continue
		}
		// Need a way to test Ieterate through
		// changes in the feature weight and
		// numBuckets.
		feat := fier.ColDef[featNdx]
		newNumBuck := feat.NumBuck
		newWeight := feat.FeatWeight * 0.5

		newPrec, kept := fier.optRunOne(featNdx, newNumBuck, newWeight, lastPrec, trainRows, testRows)
		fmt.Printf("newPrec=%v  kept=%v", newPrec, kept)

		elap := Nowms() - startTime
		fmt.Printf(" elap=%v", elap)
		if elap > maxTimeSec {
			break
		}

	} // for
}
