// Methods to apply the QProb Classifier for 1..N
// records and return results in a form that can
// easily be consumed by downstream consumers
// such as the optimizer and classifyFiles

// classifyResult.go
package qprob

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

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
	ActClass  int16 // actual class when known -9999 when not known
	Classes   map[int16]ResultItem
	Features  []ResultForFeat
}

// Structures to support most basic
// classify save of chosen class and
// basic probability for that choice
type SimpResRow struct {
	BestClass int16
	BestProb  float32
	ActClass  int16
}

type SimpResults struct {
	TotCnt int32
	SucCnt int32
	Precis float32
	Rows   []SimpResRow
}

// Save simple results as if running validation test
func (sr *SimpResults) AsStrToBuffTest(sb *bytes.Buffer) {
	fmt.Fprintln(sb, "ndx,bestClass,bestProb,actClass,status")
	for ndx, row := range sr.Rows {
		stat := "ok"
		if row.BestClass != row.ActClass {
			stat = "fail"
		}
		fmt.Fprintf(sb, "%v,%v,%v,%v,%s\n",
			ndx, row.BestClass, row.BestProb, row.ActClass, stat)
	}

}

// Save simple results as if classifying request
func (sr *SimpResults) AsStrToBuffClass(sb *bytes.Buffer) {
	fmt.Fprintln(sb, "ndx,bestClass,bestProb")
	for ndx, row := range sr.Rows {

		fmt.Fprintf(sb, "%v,%v,%v\n",
			ndx, row.BestClass, row.BestProb)
	}
}

func (sr *SimpResults) ToDispStr() string {
	var sbb bytes.Buffer
	sb := &sbb
	sr.AsStrToBuffTest(sb)
	failCnt := sr.TotCnt - sr.SucCnt
	failP := 1.0 - sr.Precis
	fmt.Fprintf(sb, "numRow=%v  sucCnt=%v precis=%v failCnt=%v failPort=%v\n",
		sr.TotCnt, sr.SucCnt, sr.Precis, failCnt, failP)
	return sb.String()
}

func (sr *SimpResults) ToJSON() []byte {
	ba, _ := json.Marshal(sr)
	return ba
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
				baseProb := float32(classCnt) / float32(buck.totCnt)
				classProb := fier.ClassProb[classId]
				//workProb := baseProb / classProb
				//fBuckWrk.Prob = workProb
				fBuckWrk.Prob = baseProb - classProb
				fwrk.TotCnt += classCnt
				fwrk.Cls[classId] = *fBuckWrk
				clswrk, clsFound := clsm[classId]
				if clsFound == false {
					clswrk = new(ResultItem)
					clsm[classId] = clswrk
				}
				// NOTE: In every case tested the precision
				//   at 100% accuracy droped when we used
				//   the baseProb * classProb  instead of
				//    simply the baseProb.
				//clswrk.Prob += workProb * feat.FeatWeight
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
		//classWrk.Prob = classWrk.Prob / float32(fier.totFeatWeight())
		classWrk.Prob = classWrk.Prob / float32(len(fier.ColDef))
		tout.Classes[classId] = *classWrk
		if bestProb < classWrk.Prob {
			bestProb = classWrk.Prob
			tout.BestProb = classWrk.Prob
			tout.BestClass = classId
			tout.ActClass = -9999
		}
	}

	return tout
} // func

/* Classify a array of rows returns the analyzed
rows and the Test summary results.  */
func (fier *Classifier) ClassifyRows(rows [][]float32) ([]ResultForRow, *SimpResults) {
	numRow := len(rows)
	tout := make([]ResultForRow, numRow)

	//sucessCnt := 0
	//rowCnt := 0
	resRows := new(SimpResults)
	resRows.TotCnt = int32(numRow)
	resRows.Rows = make([]SimpResRow, numRow)

	for ndx := 0; ndx < numRow; ndx++ {
		rowIn := rows[ndx]

		cres := fier.ClassRow(rowIn)
		cres.ActClass = int16(rowIn[fier.ClassCol])

		// Copy into Simplified structure
		// for use generating the output
		// CSV.   We also need this one to
		// generate the simplified version
		// of the JSON
		rrow := &resRows.Rows[ndx]
		rrow.BestClass = cres.BestClass
		rrow.BestProb = cres.BestProb
		rrow.ActClass = cres.ActClass
		if rrow.BestClass == rrow.ActClass {
			resRows.SucCnt += 1
		}
		//if cres.actClass == cres.BestClass {
		//	sucessCnt += 1
		//}
		// TODO: We want to track sucess by class
		// TODO: Build the Result Records here
		// TODO: Return them as a separate set of parms
		//rowCnt += 1

	} // for row
	resRows.Precis = float32(resRows.SucCnt) / float32(resRows.TotCnt)
	//percCorr := (float32(sucessCnt) / float32(rowCnt)) * float32(100.0)
	//percFail := 100.0 - percCorr

	return tout, resRows

}

// NOTE: Consider just writing the formatting from JSON results
//   save the JSON results and make it easily read by ajax
//   That would save writting custom formatting in go and push
//   over to javascript where it is easier.

// function save testResult by row as csv

// function save classifyResult by row as csv

// function save results by class as json

// function save summary results by class as csv

// function printout nice summary of results by class
