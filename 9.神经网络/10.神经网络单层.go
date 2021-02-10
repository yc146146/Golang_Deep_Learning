package main

import "fmt"

type MulLayerFloat struct {
	x float64
	y float64
}

//正向传播
func (this *MulLayerFloat)forward(x,y float64)float64{
	this.x = x
	this.y = y
	out := x*y
	return  out
}

//反向传播
func (this *MulLayerFloat)backward(dout float64)(float64,float64){
	dx := dout*this.y
	dy := dout*this.x

	return dx,dy
}

func TestRunMulLayerFloat(){
	apple := 100.0
	appleNum := 2.0
	tax := 1.1

	mulAppleLayer := &MulLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}

	//数据传播
	applePrice := mulAppleLayer.forward(apple, appleNum)
	Price := mulTaxLayer.forward(applePrice, tax)

	fmt.Println(Price)

	dprice := 1.0
	dapplePrice,dtax := mulTaxLayer.backward(dprice)
	dapple, dappleNum := mulAppleLayer.backward(dapplePrice)
	fmt.Println(dapplePrice,dapple, dappleNum, dtax)

}




func main() {

	TestRunMulLayerFloat()

}
