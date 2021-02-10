package main

import (
	"fmt"
	"gorgonia.org/tensor"
)

type Relu struct {
	mask[]int
}

//正向传播
func (this * Relu)Forward(x *tensor.Dense)*tensor.Dense{
	it := x.Iterator()
	for !it.Done(){
		idx, _ := it.Next()
		a := x.Get(idx).(float64)
		if a<0{
			x.Set(idx, 0.0)
			this.mask=append(this.mask, idx)
		}
	}
	return x
}


//反向传播
func (this * Relu)Backward(dout *tensor.Dense)*tensor.Dense{

	for _,idx := range this.mask{
		dout.Set(idx, 0)
	}

	return dout
}

func main() {
	x := tensor.New(tensor.WithShape(2,2), tensor.WithBacking([]float64{1.0,-0.5,-2.0,3.0}))

	relu := &Relu{}
	relu.Forward(x)
	fmt.Println(x)

}
