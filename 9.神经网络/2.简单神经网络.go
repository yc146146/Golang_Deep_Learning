package main

import (
	"fmt"
	"math"
)

func StepFunc(x float64)int{
	if x>0{
		return 1
	}else{
		return 0
	}
}

//逻辑回归函数
func sigmoid(x float64)float64{
	return 1.0/(1.0+math.Exp(-x))
}

//取最大
func relu(x float64)float64{
	return math.Max(0,x)
}

func main() {
	fmt.Println("data")
	for _,i:=range []float64{-5.0,5.0,0.1}{
		fmt.Printf("%v->%v\n", i, StepFunc(i))
	}

	fmt.Println()
	fmt.Println("sigmoid")
	for _,i:=range []float64{-5.0,5.0,0.1}{
		fmt.Printf("%v->%v\n", i, sigmoid(i))
	}

	fmt.Println()

	fmt.Println()
	fmt.Println("relu")
	for _,i:=range []float64{-5.0,5.0,0.1}{
		fmt.Printf("%v->%v\n", i, relu(i))
	}

	fmt.Println()

}
