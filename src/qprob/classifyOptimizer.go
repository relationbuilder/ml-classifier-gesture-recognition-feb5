// classifyOptimizer.go
package qprob

import (
	"fmt"
)

type optFeatSet struct {
	numQuant int16
	priority float32
}

type optSettings struct {
	features []optFeatSet // settings for each features
}

type optResByClass struct {
}

type optResult struct {
}

func testClassifyOpt() {
	fmt.Println("Hello World!")
}
