package main

import (
	"fmt"
)


func And(x1,x2 float64)int{
	x := []float64{x1, x2}
	w:=[]float64{0.5,0.5}
	b:=-0.50001

	tmp := x[0]*w[0]+x[1]*w[1]+b
	fmt.Println(tmp)
	if tmp<=0{
		return 0
	}else{
		return 1
	}
}

func Or(x1,x2 float64)int{
	x := []float64{x1, x2}
	w:=[]float64{0.5,0.5}
	b:=-0.00001

	tmp := x[0]*w[0]+x[1]*w[1]+b
	fmt.Println(tmp)
	if tmp<=0{
		return 0
	}else{
		return 1
	}
}


func main() {
	tests := [][]float64{
		[]float64{0,0},
		[]float64{1,0},
		[]float64{0,1},
		[]float64{1,1},


	}

	fmt.Println(tests)
	for _,xs := range tests{
		//fmt.Println(xs, And(xs[0],xs[1]))
		fmt.Println(xs, Or(xs[0],xs[1]))
	}


}
