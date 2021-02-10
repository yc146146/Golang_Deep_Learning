package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type FirstNetWork struct {
	w1,b1,w2,b2,w3,b3 *mat.Dense
}

//逻辑回归函数
func sigmoid(x float64)float64{
	return 1.0/(1.0+math.Exp(-x))
}


func initNetWork()*FirstNetWork{
	return &FirstNetWork{
		w1:mat.NewDense(2,3,[]float64{
			0.1,0.3,0.5,
			0.2,0.4,0.6,
		}),
		b1:mat.NewDense(1,3,[]float64{0.1,0.2,0.3}),
		w2:mat.NewDense(3,2,[]float64{
			0.1,0.4,
			0.2,0.5,
			0.3,0.6,
		}),
		b2:mat.NewDense(1,2,[]float64{0.1,0.2}),
		w3:mat.NewDense(2,2,[]float64{
			0.1,0.2,0.3,0.4,

		}),
		b3:mat.NewDense(1,2,[]float64{0.1,0.2}),
	}
}

func sigmoidDense(a* mat.Dense)*mat.Dense{
	var z mat.Dense
	z.Apply(func(i,j int,v float64)float64{
		return sigmoid(v)
	},a)

	return &z
}

func identityDensen(a *mat.Dense)*mat.Dense{
	var z mat.Dense
	z.Apply(func(i,j int,v float64)float64{
		return v
	},a)

	return &z
}

func (n* FirstNetWork)forward(x*mat.Dense)*mat.Dense{
	var a1 mat.Dense
	a1.Mul(x,n.w1)
	a1.Add(&a1, n.b1)

	var z1 *mat.Dense
	z1=sigmoidDense(&a1)
	fmt.Println("a1",a1)
	fmt.Println("z1",z1)


	var a2 mat.Dense
	a2.Mul(z1,n.w2)
	a2.Add(&a2, n.b2)

	var z2 *mat.Dense
	z2=sigmoidDense(&a2)
	fmt.Println("a2",a2)
	fmt.Println("z2",z2)

	var a3 mat.Dense
	a3.Mul(z2,n.w3)
	a3.Add(&a3, n.b3)

	var y *mat.Dense

	y = SoftMax(&a3)
	fmt.Println("a3",a3)
	fmt.Println("y",y)

	//y存储最大概率
	return y

}

func SoftMax(a *mat.Dense)*mat.Dense{
	y := mat.DenseCopyOf(a)
	c := mat.Max(y)

	y.Apply(func(i,j int, v float64) float64{
		return math.Exp(v-c)
	},y)
	sum := mat.Sum(y)

	y.Apply(func(i,j int, v float64) float64{
		return v/sum
	},y)

	return y

}

func Train(){
	x := mat.NewDense(1,2, []float64{1.0,0.5})
	n:=initNetWork()
	y := n.forward(x)
	fmt.Println(y)
	fmt.Println(n.w1, n.b1)
	fmt.Println(n.w2, n.b3)
	fmt.Println(n.w3, n.b3)


}


func main() {
	Train()

	x := mat.NewDense(1,2, []float64{0.8,0.4})
	n:=initNetWork()
	y := n.forward(x)
	fmt.Println(y)
	fmt.Println(n.w1, n.b1)
	fmt.Println(n.w2, n.b3)
	fmt.Println(n.w3, n.b3)


}

