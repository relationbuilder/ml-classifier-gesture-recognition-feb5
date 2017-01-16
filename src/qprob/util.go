package qprob

import (
	"math"
	"strconv"
	s "strings"

	//"bufio"
	//"fmt"
	//"io"
	//"io/ioutil"
	//"log"
	//"net/http"
	//"os"
	//"os/exec"
)

// TODO: Move these to a common utility
// module.
func MaxI16(x, y int16) int16 {
	if x > y {
		return x
	} else {
		return y
	}
}

func MinI16(x, y int16) int16 {
	if x < y {
		return x
	} else {
		return y
	}
}

func MaxI32(x, y int32) int32 {
	if x > y {
		return x
	} else {
		return y
	}
}

func MinI32(x, y int32) int32 {
	if x < y {
		return x
	} else {
		return y
	}
}

func MaxF32(x, y float32) float32 {
	if x > y {
		return x
	} else {
		return y
	}
}

func MinF32(x, y float32) float32 {
	if x < y {
		return x
	} else {
		return y
	}
}

func ParseStrAsArrInt32(astr string) []int32 {
	a := s.Split(astr, ",")
	numCol := len(a)
	wrkArr := make([]int32, numCol)
	for fc := 0; fc < numCol; fc++ {
		ctxt := s.TrimSpace(a[fc])
		i64, err := strconv.ParseInt(ctxt, 10, 32)
		i32 := int32(i64)
		if err != nil {
			i32 = math.MaxInt32
		}
		wrkArr[fc] = i32
	}
	return wrkArr
}

/* Any values that failed to parse will contain
math.MaxFloat32 as error indicator */
func ParseStrAsArrFloat(astr string) []float32 {
	a := s.Split(astr, ",")
	numCol := len(a)
	wrkArr := make([]float32, numCol)
	for fc := 0; fc < numCol; fc++ {
		ctxt := s.TrimSpace(a[fc])
		f64, err := strconv.ParseFloat(ctxt, 32)
		f32 := float32(f64)
		if err != nil {
			f32 = math.MaxFloat32
		}
		wrkArr[fc] = f32
	}
	return wrkArr
}
