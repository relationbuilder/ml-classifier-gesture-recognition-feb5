/* splitArray.go  Utility to split an array into two parts.  One used for
testing the other used for training.
(C) Joe Ellsworth Jan-2017 License MIT See license.txt */
package qutil

// Divide the input array into two output arrays
// main which contains most of the records and
// auxOut which receives one row for every
// oneEvery rows in main.  The first skipNumFirst
// rows always go to main before aux gets any.
// This is normally used to isolate a set of data
// between training and test but is also used to
// isolate rows for the optimizer
// skipNumFirst is used to shuffle sets to allow
// easy extraction of different records to vary
// the test set.
func SplitFloatArrOneEvery(ain [][]float32, skipNumFirst int, oneEvery int) ([][]float32, [][]float32) {
	numRow := len(ain)
	auxNumRow := (numRow - skipNumFirst) / oneEvery
	if auxNumRow < 1 {
		auxNumRow = 1
	}
	mainNumRow := numRow - auxNumRow
	if mainNumRow < 1 {
		mainNumRow = 1
	}

	mOut := make([][]float32, 0, mainNumRow+2)
	auxOut := make([][]float32, 0, auxNumRow+2)
	keepCnt := 0
	for ndx, row := range ain {
		if ndx >= skipNumFirst && keepCnt >= oneEvery {
			auxOut = append(auxOut, row)
			keepCnt = 0
		} else {
			mOut = append(mOut, row)
			keepCnt += 1
		}
	} // for
	return mOut, auxOut
}

// TODO:  We will need a version of this for time series data
//  that composes the Aux Array out of only rows at the end
//  of the series.
