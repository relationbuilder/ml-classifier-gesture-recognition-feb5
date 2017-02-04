// classifyRequest.go
package qprob

import (
	"encoding/json"
	//"fmt"
)

type ClassifyRequest struct {
	TrainInFi  string
	TestInFi   string
	ClassInFi  string
	ClassOutFi string
	MaxNumBuck int16
	ModelFi    string

	TestOutFi string

	DoOpt           bool
	DoTest          bool
	DoClassify      bool
	LoadModel       bool
	OkToRun         bool
	WriteJSON       bool
	WriteCSV        bool
	WriteDetails    bool
	WriteFullCSV    bool
	DetToStdOut     bool
	OptPreRandomize bool
	OptMaxTime      float64
	OptClassId      int16
	OptMinRecall    float32
	OptMaxPrec      float32

	DoPreAnalyze    bool
	AnalClassId     int16
	AnalMinRecall   float32
	AnalMinPrec     float32
	AnalSeekOptFeat bool
	AnalSplitType   int16   // 1 = take from body,  2 = split end
	AnalTestPort    float32 // portion of training set to use for test data

	Header string
}

func (req *ClassifyRequest) ToJSON() string {
	aStr, _ := json.Marshal(req)
	return string(aStr)
}

func MakeEmptyClassifyFilesRequest() *ClassifyRequest {
	tout := new(ClassifyRequest)
	tout.OkToRun = true
	// TODO: Lets initialize those we need
	return tout
}
