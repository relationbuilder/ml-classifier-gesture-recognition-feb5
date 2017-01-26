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

type testResByClass struct {
	classId      int16
	classCnt     int32
	classProb    float32
	foundCount   int32
	recall       float32
	foundCorrect int32
	rows         []testRes // This is left nil unless specifically requested.
	lift         float32
}

type summaryResult struct {
	byClass    []testResByClass
	cntTotal   int32
	cntCorrect int32
	precision  float32
	byRow      []testRowRes
}

func TestClassifyAnal() {
	fmt.Println("Hello World!")
}

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
