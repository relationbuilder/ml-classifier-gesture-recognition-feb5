package main

/* classifyFile.go  - Train from 1 CSV File and classify using
using a second test file.  Output statistics on classification
from algorithm versus actual classification */

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"qprob"
	"strconv"
	s "strings"
)

func check(msg string, e error) {
	if e != nil {
		fmt.Println("ERROR:")
		fmt.Println(e)
		panic(e)
	}
}

func printLn(f *os.File, txt string) {
	_, err1 := f.WriteString(txt)
	check("err in printLn ", err1)

	_, err2 := f.WriteString("\n")
	check("err in printLn ", err2)
}

/*
  Uses the training file to train the classifier
  then reads Reads lines out of test CSV file
  having the classifier classify each line
  and report how accurate the classifier is.
  reports both precision and recall by class.
*/
func ClassifyFiles(trainFiName string, testFiName string, numBuck int16) {
	var sbb bytes.Buffer
	sb := &sbb

	fmt.Fprintf(sb, "\rClassifyFiles\r  trainFiName=%s\r  testFiName=%s\r  numBuck=%v\n",
		trainFiName, testFiName, numBuck)

	fier := qprob.LoadClassifierTrainFile(trainFiName, "test", numBuck)
	fmt.Fprintln(sb, "constructor complete")
	//fmt.Println(fier.String())

	fmt.Fprintln(sb, "\nfinished build Now Try to classify")
	fiIn, err := os.Open(testFiName)
	check("opening file", err)
	if err != nil {
		log.Fatal(err)
	}
	scanner := bufio.NewScanner(fiIn)
	defer fiIn.Close()

	// Copy of header to both files
	scanner.Scan() // skip headers
	//headTxt := s.TrimSpace(scanner.Text())

	fmt.Fprintln(sb, "row,predClass,predProb,actClass,status\n")
	// Copy the rows.
	sucessCnt := 0
	rowCnt := 0
	for scanner.Scan() {
		txt := s.TrimSpace(scanner.Text())

		if err := scanner.Err(); err != nil {
			log.Fatal(err)
		}

		cres := fier.ClassRowStr(txt)
		flds := qprob.ParseStrAsArrInt32(txt)
		actualClass := int16(flds[fier.ClassCol])
		statTxt := "sucess"
		if actualClass != cres.BestClass {
			statTxt = "fail  "
		} else {
			sucessCnt += 1
		}
		// TODO: We want to track sucess by class

		fmt.Fprintf(sb, "%v,%v,%v,%v,%v\n",
			rowCnt, cres.BestClass, cres.BestProb, actualClass, statTxt)

		rowCnt += 1
	} // for row

	percCorr := (float32(sucessCnt) / float32(rowCnt)) * float32(100.0)
	percFail := 100.0 - percCorr

	fmt.Fprintf(sb, "tested %v rows  with %v correct sucess=%v%% fail=%v%%",
		rowCnt, sucessCnt, percCorr, percFail)
	sb.WriteTo(os.Stdout)
}

func printHelp() {
	fmt.Println("eg: classifyFiles data/titanic.train.csv data/titanic/test.csv 6")
	fmt.Println("will read training file to build classifer")
	fmt.Println("will read test data file to classify each line")
	fmt.Println("will report results of test versus repored class")
	fmt.Println("classifier will use numBuck buckets.")
	fmt.Printf("args = %v\n", os.Args)
}

func main() {
	fmt.Println("classifyFiles.go trainFiName TestFiName numBuck")
	if len(os.Args) != 4 {
		fmt.Println("ERROR: Incorrect number of command args")
		printHelp()
		return
	}

	trainFiName := s.TrimSpace(os.Args[1])
	if _, err := os.Stat(trainFiName); os.IsNotExist(err) {
		fmt.Printf("ERROR: train file does not exist %s\n", trainFiName)
		printHelp()
		return
	}

	testFiName := s.TrimSpace(os.Args[2])
	if _, err := os.Stat(testFiName); os.IsNotExist(err) {
		fmt.Printf("ERROR: test file does not exist %s\n", testFiName)
		printHelp()
		return
	}

	numBuck, err := strconv.ParseInt(os.Args[3], 10, 32)
	if err != nil {
		fmt.Printf("ERROR: parsing tmpEvery valIn=%v\n", os.Args[2])
		printHelp()
		return
	}

	ClassifyFiles(trainFiName, testFiName, int16(numBuck))
}
