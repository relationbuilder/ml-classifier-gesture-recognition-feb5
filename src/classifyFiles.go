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
	"qutil"
	//"strconv"
	"encoding/json"
	"io/ioutil"
	s "strings"
)

type ClassifyFilesRequest struct {
	TrainInFi  string
	TestInFi   string
	ClassInFi  string
	ClassOutFi string
	NumBuck    int16
	ModelFi    string

	TestOutFi string

	DoOpt        bool
	DoTest       bool
	DoClassify   bool
	LoadModel    bool
	OkToRun      bool
	WriteJSON    bool
	WriteCSV     bool
	WriteDetails bool
	WriteFullCSV bool
	DetToStdOut  bool
}

func (req *ClassifyFilesRequest) toJSON() string {
	aStr, _ := json.Marshal(req)
	return string(aStr)
}

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

func MakeEmptyClassifyFilesRequest() *ClassifyFilesRequest {
	tout := new(ClassifyFilesRequest)
	tout.OkToRun = true
	// TODO: Lets initialize those we need
	return tout
}

func LoadCSVRows(fiName string) [][]float32 {
	rows := make([][]float32, 0, 1)
	fiIn, err := os.Open(fiName)
	check("opening file", err)
	if err != nil {
		log.Fatal(err)
	}
	scanner := bufio.NewScanner(fiIn)
	defer fiIn.Close()

	// Copy of header to both files
	scanner.Scan() // skip headers
	//headTxt := s.TrimSpace(scanner.Text())

	for scanner.Scan() {
		txt := s.TrimSpace(scanner.Text())

		if err := scanner.Err(); err != nil {
			log.Fatal(err)
		}

		flds := qprob.ParseStrAsArrFloat32(txt)
		rows = append(rows, flds)

	} // for row
	return rows
}

func ProcessRowsRows(fier *qprob.Classifier, req *ClassifyFilesRequest, rows [][]float32, outBaseName string, asTest bool) {
	fmt.Println("\nfinished build Now Try to classify")
	detRows, sumRows := fier.ClassifyRows(rows)

	//
	fmt.Printf("num deRows=%v\n", len(detRows))
	fmt.Printf("num sumRows.Precis=%v\n", sumRows.Precis)

	if req.WriteJSON {
		jsonsumstr := sumRows.ToJSON()
		outFileName := s.Replace(outBaseName, ".csv", ".out.sum.json", -1)
		outFileName = s.Replace(outFileName, ".out.out", ".out", -1)
		fmt.Printf("write JSON sum rows to %s\n", outFileName)
		if req.DetToStdOut {
			fmt.Printf("sumRows asJSON=%s\n", jsonsumstr)
		}
		ioutil.WriteFile(outFileName, jsonsumstr, 0644)

		// Add class probability output
		// add detailed probability output
	}

	// Convert the summary Rows into printable Output to display
	if req.WriteCSV {
		var sbb bytes.Buffer
		sb := &sbb
		if asTest {
			sumRows.AsStrToBuffTest(sb)
		} else {
			sumRows.AsStrToBuffClass(sb)
		}
		outFileName := s.Replace(outBaseName, ".csv", ".out.sum.csv", -1)
		outFileName = s.Replace(outFileName, ".out.out", ".out", -1)
		fmt.Printf("write CSV sum rows to %s\n", outFileName)
		ioutil.WriteFile(outFileName, sb.Bytes(), 0644)
		if req.DetToStdOut {
			fmt.Printf("As Disp Str\n%s\n", sb.String())
			failCnt := sumRows.TotCnt - sumRows.SucCnt
			failP := 1.0 - sumRows.Precis
			fmt.Printf("numRow=%v  sucCnt=%v precis=%v failCnt=%v failPort=%v",
				sumRows.TotCnt, sumRows.SucCnt, sumRows.Precis, failCnt, failP)
		}

		// add class probability output

		// add detailed probability output
	}
}

/*
  Uses the training file to train the classifier
  then reads Reads lines out of test CSV file
  having the classifier classify each line
  and report how accurate the classifier is.
  reports both precision and recall by class.

  NOTE: Moved to processingRows for Test because
   I wanted that methd for use with the optimizer
   not quite as scalable as the line by line but
   we also needed different options for output.
*/
func ClassifyTestFiles(req *ClassifyFilesRequest) {

	fmt.Printf("\rClassifyFiles\r  trainFiName=%s\r  testFiName=%s\r  numBuck=%v\n",
		req.TrainInFi, req.TestInFi, req.NumBuck)

	fier := qprob.LoadClassifierTrainFile(req.TrainInFi, "test", req.NumBuck)
	fmt.Println("constructor complete")
	//fmt.Println(fier.String())

	// If we were processing a test file then
	// save the output to represent it's results
	if req.TestInFi != "" {
		fmt.Printf("processing test data %s", req.TestInFi)
		testRows := LoadCSVRows(req.TestInFi)
		fmt.Printf("Loaded %v rows\n", len(testRows))
		ProcessRowsRows(fier, req, testRows, req.TestOutFi, true)
	}

	// If we have a classification job then process it.
	if req.ClassInFi != "" {
		fmt.Printf("processing classify data %s", req.ClassInFi)
		rows := LoadCSVRows(req.ClassInFi)
		fmt.Printf("Loaded %v rows\n", len(rows))
		ProcessRowsRows(fier, req, rows, req.ClassOutFi, false)
	}

}

/*
  Uses the training file to train the classifier
  then reads Reads lines out of test CSV file
  having the classifier classify each line
  and report how accurate the classifier is.
  reports both precision and recall by class.
*/
func ClassifyTestFilesLargeFile(req *ClassifyFilesRequest) {
	var sbb bytes.Buffer
	sb := &sbb
	trainFiName := req.TrainInFi
	testFiName := req.TestInFi
	numBuck := req.NumBuck

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
	msg := `
    -train=finame      file containing training data
                       optional when model input is specified
  
    -test=finame       file containing data to use to test model
	                   file must exist when specified. 
                       optional when -class is specified. 
					
    -class=finame      name of file containing data to classify 
	                   must exist is specified.   Optional when
					   -test is specified.  By convention class
					   is set to -1 in input class files but the
					   system doese not care. 
	
	-classout=finame   name of file to write classify results to
	                   will be written in csv format.  If not
					   specified default name  will be name 
					   specified by -class with .csv
					   replaced with .out.csv.  By convention
					   all named output files should end with
					   .out.csv. 
	
	-testout=finame    Write test output CSV file name to this file
	                   instead of the default output file.   By convention
					   all output files should end with .out.csv
					
	-numBuck=10        Number of qanta buckets to use by default
	                   for the model but the optimizer may change
					   this on a feature by feature basis 
	
	-writeJSON=true    if present then write results to JSON files
	                   otherwise will only generate CSV.
					
	-writeCSV=true     Will write output in CSV form which will 
	                   require multiple files in some instances
					   or supress some explanatory information
					   defaults to true if not specified.
					
	-writeFullcsv=false Write the original CSV with all columns
	                   the same except for the class column values
					   will be changed to the predicted class 
					   defaults to false.
					
    -writeDetails=true Write files containing detailed probability
	                   by row in addition to the summary information
					   this shows the probability of each row belonging
					   to each class. 
					   file extensions will be .det added to path 
					   name. 
					
    -detToStdOut=false When true will print values saved in the generated
	                   files to stdout as things are processed.  This consumes
					   considerable time so turn of except when debugging. 
					   defaults to true. 

  eg: classifyFiles -train=data/titanic.train.csv -test=data/titanic/test.csv -numBuck=6
	 will read training file to build classifer
	 will read test data file to classify each line
     will report results of test versus repored class
	 classifier will use numBuck buckets.
	 classified test records will be written to data/titanic/test.out.csv 
	 classified test summary will be written to data/titanic/test.meta.txt
	
	
  eg: classifyFiles -train=data/titanic.train.csv -class=data/toclassify.csv")
	 Will read training file data/titanic.train.csv  build the model
	 and will read file data/toclassify.csv as records to classify.
	 The resulting output will be written to data/toclassify.out.csv 
	 because no -classout was specified.  Numuck defaulted to 10 because
	 was not specified.
	
   It is legal to specify both -class and -test it will run the test 
   and then run the classify.

   It is legal to use a filter generated as a -test input file as input
   to the -class option.   
					
    -------------------
	---- FUTURE -------
    -------------------
	
					
	-readModel=true   If defined and set to true and when -model is set
	                   the exisitng model file will be read before the
					   training file.  Otherwise the model file will be
					   ignored and replaced with data from the training
					   phase.  Defaults to true if not specified
					
	-model             Name of file to read as the model from. It will
	                   be read if the model before the training data if
					   it exists.  If File does not exist it will be 
					   generated when the training has been completed
					   and after optimization passes.
					
		
	
					
    -optMinPrec=95     Will run optimizer cycles until a minimum 
	                   precision at 100% recall has been reached.
					   will stop when optimizer has ran optMaxCycles
					   if not set then no optimizer is ran. 
	
    -optMaxTime=1      Max Time each cycle of Optimizer is allowed 
	                   to run when seeking to reach optMinPrec
					   Specified as integer num representing 
					   seconds between 1 and 100. Defaults to 
					   10 if not specified. 
					
	-optMax=10         Max number of cyles the optimizer is allowed
	                   to run between 1 and 1000 when seekng to
					   reach specified precision. Defaults to 
					   10 if not set.  Values less than 1 or 
					   greater than 1000 will be reset. 
	
	-optOKBuck=true    If true then optimizer is allowed to 
	                   change numer of quant buckets.  Defaults
					   to true if not set.  Must be true if
					   -optOKWeight is false when he optimizer 
					   is set. 
					
	-optOKWeight       If True the optimizer is allowed to change
	                   feature weight.  Defaults to true if not set.
					   Must be true if -optOKBuck is false when
					   optimizer is ran. 
					
	
  -`
	fmt.Println(msg)
}

func checkParms(msg string, abort bool) {
	if abort {
		fmt.Printf("ERROR: %s", msg)
		printHelp()
		log.Fatal("Exit")
	}
}

func defCSVOutName(str string) string {
	ts := str
	ts = s.Replace(ts, ".csv", ".out.csv", -1)
	return ts
}

func defModelName(str string) string {
	ts := str
	ts = s.Replace(ts, "csv", ".model.csv", -1)
	return ts
}

func ParseCommandParms(args []string) *ClassifyFilesRequest {
	aReq := MakeEmptyClassifyFilesRequest()
	parms := qutil.ParseCommandLine(args)
	fmt.Println(parms.String())

	aReq.TrainInFi = parms.Sval("train", "")
	aReq.TestInFi = parms.Sval("test", "")
	aReq.ClassInFi = parms.Sval("class", "")
	aReq.ModelFi = parms.Sval("model", aReq.TrainInFi)
	aReq.ClassOutFi = parms.Sval("classout", defCSVOutName(aReq.ClassInFi))
	aReq.TestOutFi = parms.Sval("testout", defCSVOutName(aReq.TestInFi))
	aReq.NumBuck = int16(parms.Ival("numuck", 10))
	aReq.LoadModel = parms.Bval("loadModel", true)
	aReq.WriteJSON = parms.Bval("writejson", false)
	aReq.WriteCSV = parms.Bval("writeCSV", true)
	aReq.WriteDetails = parms.Bval("writedetails", false)
	aReq.WriteFullCSV = parms.Bval("writefullcsv", false)
	aReq.DetToStdOut = parms.Bval("dettostdout", true)
	aReq.OkToRun = false

	if aReq.TrainInFi == "" && aReq.ModelFi == "" {
		checkParms("Either training file or model file must be specified", true)
		return aReq
	}

	if aReq.TrainInFi != "" {
		if _, err := os.Stat(aReq.TrainInFi); os.IsNotExist(err) {
			fmt.Printf("ERROR: train file does not exist %s\n", aReq.TrainInFi)
			printHelp()
			return aReq
		}
	}

	if aReq.TestInFi != "" {
		if _, err := os.Stat(aReq.TestInFi); os.IsNotExist(err) {
			fmt.Printf("ERROR: test file does not exist %s\n", aReq.TestInFi)
			printHelp()
			return aReq

		}
	}

	if aReq.ClassInFi != "" {
		if _, err := os.Stat(aReq.ClassInFi); os.IsNotExist(err) {
			fmt.Printf("ERROR: Class file does not exist %s\n", aReq.ClassInFi)
			printHelp()
			return aReq

		}
	}

	if aReq.ModelFi != "" && aReq.LoadModel == true {
		if _, err := os.Stat(aReq.ModelFi); os.IsNotExist(err) {
			fmt.Printf("ERROR: model file does not exist %s\n", aReq.ModelFi)
			printHelp()
			return aReq
		}
	}
	aReq.OkToRun = true
	return aReq

}
func main() {
	fmt.Println("classifyFiles.go ")
	req := ParseCommandParms(os.Args)
	fmt.Printf("parsed commands %s\n", req.toJSON())

	if req.OkToRun {
		fmt.Println("start ClassifyTestFiles()")
		ClassifyTestFiles(req)
		fmt.Println("Finished ClassifyTestFiles()")
	} else {
		fmt.Println("Can not run problem with input parms")
	}
}
