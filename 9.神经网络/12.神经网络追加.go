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

type AddLayerFloat struct {

}

//向前传播
func (this *AddLayerFloat)forward(x,y float64)float64{
	out := x+y
	return out
}

//向后传播
func (this *AddLayerFloat)backward(dout float64)(float64, float64){
	dx := dout*1.0
	dy := dout*1.0
	return dx, dy
}

func testMutiLayer(){
	apple := 100.0
	appleNum := 2.0
	orange := 150.0
	orangeNum := 3.0
	tax :=1.1

	//构造多层神经网络
	mulAppleLayer := &MulLayerFloat{}
	mulOrangeLayer := &MulLayerFloat{}
	addappleorange := &AddLayerFloat{}
	mulTaxLayer := &MulLayerFloat{}


	//正向传播
	applePrice := mulAppleLayer.forward(apple, appleNum)
	orangePrice := mulOrangeLayer.forward(orange, orangeNum)
	allprice := addappleorange.forward(applePrice, orangePrice)
	price := mulTaxLayer.forward(allprice, tax)

	//	反向传播
	dprice := 1.0
	dallprice, dtax := mulTaxLayer.backward(dprice)
	dappleprice,dorangeprice := addappleorange.backward(dallprice)
	dorange, dorangeNum := mulOrangeLayer.backward(dorangeprice)
	dapple, dappleNum := mulAppleLayer.backward(dappleprice)

	//查看价格
	fmt.Println(price)
	fmt.Println(dapple, dappleNum, dorange,dorangeNum, dtax)

}






func main() {

	testMutiLayer()

}
