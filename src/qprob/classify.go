package qprob

//"fmt"
//"io"
//"io/ioutil"
//"log"
//"net/http"
//"os"
//"os/exec"
//"strings"

const (
	QTBucket = 1 // Quantize using bucket logic
	QTInt    = 2 // Quantize with simple int index
	QTText   = 3 // Quantize as tokenized text field
)

type QuantType uint8

type FeatureDesc struct {
	enabled   bool
	colName   string
	procType  QuantType
	absMaxVal float32
	absMinVal float32
	effMaxVal float32
	effMinVal float32
	buckSize  float32
	colSpec   CSVInfo
}

// containst the count for each class
// identfied as having at least one row.
// eg for bucket 10 we have 4 classes.
// each id=1, count=10, id=5, count=4
// totalCnt is the sum of the Counts
// for all classes
type QuantList struct {
	Counts map[int32]int32
	totCnt int32
}

// includes the data and descriptive meta
// data for 1 feature of data or in CSV this
// would be 1 column of data.
type Feature struct {
	desc    FeatureDesc
	buckets map[int32]QuantList
	totCnt  int
}

// Includes the metadaa and classifer
// data for a given classifer
type QuantProbClassifier struct {
	colDef       map[int]*Feature
	colNumByName map[string]int
	colByName    map[string]*Feature
}

// A set of classifiers indexed by
// name to allow a single process
// to serve data from multiple training
// data sets
type Classifiers struct {
	classifiers map[string]*QuantProbClassifier
	xx          CSVInfo
}
