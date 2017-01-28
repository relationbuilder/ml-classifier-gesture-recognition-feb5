// classifyOptimizer.go
package qprob

import (
	"fmt"
	rand "math/rand"
	"qutil"
)

const OptMaxBuck = int16(500)
const OptMaxWeight = float32(50.0)

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

func addFeatNdx(targ []int16, aval int16, classCol int, numCol int) []int16 {
	if aval != int16(classCol) && aval < int16(numCol) {
		targ = append(targ, aval)
	}
	return targ
}

// Produce a semi random list of optimizer features
// we can ieterate across when choosing features to
// change.  It is only semi-random because we do want
// to force every feature to tested.
func (fier *Classifier) MakeOptFeatList(targLen int) []int16 {
	seed := int64(Nowms())
	fmt.Printf("mak, seedeFeatSemiRand seed=%v\n")
	r := rand.New(rand.NewSource(seed))
	tout := make([]int16, 0, targLen)
	numCol := len(fier.ColDef)
	maxCol := numCol - 1
	classCol := fier.ClassCol

	for {
		for off := 0; off < numCol; off++ {
			maxOff := maxCol - off
			if maxOff < 5 {
				maxOff = numCol
			}
			if off < 5 {
				off = numCol
			}
			//fmt.Printf("off=%v maxOff=%v\n", off, maxOff)
			rr := int16(r.Int31n(int32(off)))
			//fmt.Printf("off=%v maxOff=%v rr=%v ", off, maxOff, rr)
			tout = addFeatNdx(tout, int16(rr), classCol, numCol)
			rr = int16(rand.Int31n(int32(off)))
			tout = addFeatNdx(tout, int16(rr), classCol, numCol)
			tout = addFeatNdx(tout, int16(off), classCol, numCol)
			rr = int16(rand.Int31n(int32(maxOff)))
			tout = addFeatNdx(tout, int16(rr), classCol, numCol)
			tout = addFeatNdx(tout, int16(maxOff), classCol, numCol)
		}
		if len(tout) >= targLen {
			break
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
	if oldWeight == newWeight && oldNumBuck == newNumBuck {
		return lastPrec, false
	}
	if oldNumBuck != newNumBuck {
		fier.RetrainFeature(featNdx, trainRows)
	}
	_, sumRows := fier.ClassifyRows(testRows)
	//fmt.Printf("lastPrec=%v  newPrec=%v\n", lastPrec, sumRows.Precis)
	keepFlg := false
	// This check is a little complex but if the precision improved
	// we definitely want to keep the change. But we still want to
	// keep the change as long as the precision didn't drop as long
	// as the complexity measured in numBuck or Weight dropped.
	if sumRows.Precis > lastPrec || (sumRows.Precis >= lastPrec && newNumBuck < oldNumBuck) {
		keepFlg = true
		currPrec = sumRows.Precis
		fmt.Printf(" keep featNdx=%v oldPrec=%v newPrec=%v oldNumBuck=%v newNumBuck=%v oldWeight=%v newWeight=%v\n",
			featNdx, lastPrec, sumRows.Precis, oldNumBuck, newNumBuck, oldWeight, newWeight)
	} else {
		//fmt.Printf(" reverse newNumBuck=%v newWeight=%v\n", newNumBuck, newWeight)
		feat.NumBuck = oldNumBuck
		feat.FeatWeight = oldWeight
		currPrec = lastPrec
		if oldNumBuck != newNumBuck {
			fier.RetrainFeature(featNdx, trainRows)
		}
	} // reverse
	return currPrec, keepFlg
}

func (fier *Classifier) optRunOneFeatOneChange(changeSel int32, featNdx int16, lastPrec float32, trainRows [][]float32, testRows [][]float32) (int16, float32) {
	// We can depend on optRunOne to reset the feature to
	// it's original value if there was no improvement.
	feat := fier.ColDef[featNdx]
	currPrec := lastPrec
	kept := false
	keepCnt := int16(0)

	switch changeSel {
	case 0:
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

	case 1:

		if feat.FeatWeight > 0.0 {
			// Try Turn feature off by setting both weight to zero and buckets to 1
			currPrec, kept = fier.optRunOne(featNdx, 1.0, 0.0, currPrec, trainRows, testRows)
			if kept {
				// If we turned it off then see if it still likes the class as a single
				currPrec, kept = fier.optRunOne(featNdx, 1.0, 1.0, currPrec, trainRows, testRows)
				keepCnt++
			}
		} else {
			// try set feat Weight to 0
			currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, OptMaxWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}
		}

	case 2:

		// Try set Weight to random number between 0 and maxWeight
		adjWeight := (OptMaxWeight * rand.Float32())
		currPrec, kept = fier.optRunOne(featNdx, feat.NumBuck, adjWeight, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}

	case 3:

		// Try Num Buck to Max Buckets
		if feat.FeatWeight > 0.0 {
			currPrec, kept = fier.optRunOne(featNdx, OptMaxBuck, feat.FeatWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}
		}

	case 4:
		// Try Num Buck to 2
		if feat.FeatWeight > 0.0 {
			currPrec, kept = fier.optRunOne(featNdx, 2, feat.FeatWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}
		}

	case 5:
		// Try Set Num Buck to 1/2 current number
		if (feat.FeatWeight > 0.0) && (feat.NumBuck > 6) {
			adjNumBuck := feat.NumBuck / 2
			currPrec, kept = fier.optRunOne(featNdx, adjNumBuck, feat.FeatWeight, currPrec, trainRows, testRows)
			if kept {
				keepCnt++
			}

		}

	case 6:
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

	case 7:
		// Try first set weight and adjNum buck to random values
		if feat.FeatWeight != 0 {
			adjNumBuck := int16(rand.Int31n(int32(255)))
			adjWeight := (OptMaxWeight * rand.Float32())
			if adjNumBuck > 2 {
				currPrec, kept = fier.optRunOne(featNdx, adjNumBuck, adjWeight, currPrec, trainRows, testRows)
				if kept {
					keepCnt++
				}

			}
		}

	default:
		// Try set numBuck  to to 2 with default weight
		currPrec, kept = fier.optRunOne(featNdx, 2, 1.0, currPrec, trainRows, testRows)
		if kept {
			keepCnt++
		}
	}
	return keepCnt, currPrec
}

func (fier *Classifier) optRunOneFeat(featNdx int16, lastPrec float32, trainRows [][]float32, testRows [][]float32) (int16, float32) {

	changeSel := rand.Int31n(4)
	currPrec := lastPrec
	keepCnt := int16(0)
	switch changeSel {
	case 1:
		// every so often try all the changes
		for trySel := 1; trySel < 9; trySel++ {
			keepCnt, currPrec = fier.optRunOneFeatOneChange(int32(trySel), featNdx, lastPrec, trainRows, testRows)
		}
	default:
		trySel := rand.Int31n(9)
		keepCnt, currPrec = fier.optRunOneFeatOneChange(trySel, featNdx, lastPrec, trainRows, testRows)
	}
	return keepCnt, currPrec
}

//func saveOptSettings()
// func recordOptSettings()
// func loadOptSettings()

// Randomise the feature settings prior to starting
// the optimizer.   Will need this for Genetic algorithm
// latter.
func (fier *Classifier) RandomizeOptSettings() {
	for _, feat := range fier.ColDef {
		feat.FeatWeight = rand.Float32() * OptMaxWeight
		feat.NumBuck = int16(rand.Int31n(int32(OptMaxBuck)))
	}
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
	fmt.Printf("\nOptProcess label=%s TrainFi=%s\n", fier.Label, fier.TrainFiName)
	startTime := Nowms()
	//classes := fier.ClassIds()
	origTrainRows := fier.GetTrainRowsAsArr(OneGig)
	splitSkipPrefix := 1
	fmt.Printf("Opt num Training rows=%v\n", len(origTrainRows))
	trainRows, testRows := qutil.SplitFloatArrOneEvery(origTrainRows, splitSkipPrefix, splitOneEvery)
	fier.Retrain(trainRows)
	keepRunning := true
	featLst := fier.MakeOptFeatList(400)
	if fier.Req.OptPreRandomize {
		fmt.Printf("OptProcess if Randomizing before running optimizer")
		fier.RandomizeOptSettings()
	}

	//fmt.Printf("OptProcess len featLst=%v", featLst)
	_, lastSum := fier.ClassifyRows(testRows)
	lastPrec := lastSum.Precis
	fmt.Printf("opt numTrainRow=%v, numTestRow=%v origPrec=%v\n ", len(trainRows), len(testRows), lastPrec)
	classCol := int16(fier.ClassCol)
	numCol := int16(len(fier.ColDef))
	currPrec := lastSum.Precis
	for keepRunning {
		for _, featNdx := range featLst {
			if featNdx == classCol || featNdx >= numCol {
				continue
			}
			// Need a way to test Ieterate through
			// changes in the feature weight and
			// numBuckets.
			//feat := fier.ColDef[featNdx]
			numKept, newPrec := fier.optRunOneFeat(featNdx, currPrec, trainRows, testRows)
			if numKept > 0 {
				//fmt.Printf("opProcess featNdx=%v oldPrec=%v newPrec=%v  numKept=%v\n", featNdx, currPrec, newPrec, numKept)
				currPrec = newPrec
			}
		} // for featNdx

		splitOneEvery -= 1
		if splitOneEvery < 1 {
			splitOneEvery = 3
		}
		splitOneEvery = 3
		splitSkipPrefix += 1
		if int16(splitSkipPrefix) > 3 {
			splitSkipPrefix = 0
		}
		trainRows, testRows = qutil.SplitFloatArrOneEvery(origTrainRows, splitSkipPrefix, splitOneEvery)

		elap := Nowms() - startTime
		fmt.Printf("optProces=%v elap=%v", maxTimeSec, elap)
		if elap > maxTimeSec {
			//trainRows, testRows = qutil.SplitFloatArrOneEvery(origTrainRows, 0, 1)
			//numKept, newPrec := fier.optRunOneFeat(featNdx, currPrec, trainRows, testRows)
			break
		}

	} // for keep running
	fier.Retrain(origTrainRows)
}
