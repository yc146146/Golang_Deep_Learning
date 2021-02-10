package main

import (
	"fmt"
	"math"
)

//逻辑回归神经网络
type LogisticRegresstion struct {
	//总数量
	N int
	//输入数量
	N_in int
	//输出的数量
	N_out int
	//权重
	W [][]float64
	//系数
	B []float64
}

//逻辑回归
func LogisticRegresstion_construct(this *LogisticRegresstion, n, nin, nout int) {
	this.N = n
	this.N_in = nin
	this.N_out = nout

	this.W = make([][]float64, nout)
	for i := 0; i < nout; i++ {
		this.W[i] = make([]float64, nin)
	}
	this.B = make([]float64, nout)
}

func LogisticRegresstion_train(this *LogisticRegresstion, x, y []int, lr float64) {
	//开辟数组
	p_y_give_x := make([]float64, this.N_out)
	dy := make([]float64, this.N_out)
	for i := 0; i < this.N_out; i++ {
		p_y_give_x[i] = 0
		for j := 0; j < this.N_in; j++ {
			//叠加权重
			p_y_give_x[i] += this.W[i][j] * float64(x[j])
		}
		//叠加数据
		p_y_give_x[i] += this.B[i]
	}
	LogisticRegresstion_softmax(this, p_y_give_x)
	for i := 0; i < this.N_out; i++ {
		dy[i] = float64(y[i]) - p_y_give_x[i]
		for j := 0; j < this.N_in; j++ {
			//叠加计算权重
			this.W[i][j] += lr * dy[i] * float64(x[j]) / float64(this.N)
		}
		this.B[i] += lr * dy[i] / float64(this.N)
	}
}

func LogisticRegresstion_softmax(this *LogisticRegresstion, x []float64) {
	var max, sum float64
	for i := 0; i < this.N_out; i++ {
		if max < x[i] {
			max = x[i]
		}
	}

	for i := 0; i < this.N_out; i++ {
		//取得概率
		x[i] = math.Exp(x[i] - max)
		sum+=x[i]
	}

	for i := 0; i < this.N_out; i++ {
		x[i] /= sum
	}

}

func LogisticRegresstion_predict(this *LogisticRegresstion, x []int, y []float64) {
	for i := 0; i < this.N_out; i++ {
		y[i] = 0

		for j := 0; j < this.N_in; j++ {
			//权重叠加计算
			y[i] += this.W[i][j] * float64(x[j])
		}

		y[i] += this.B[i]
	}

	LogisticRegresstion_softmax(this, y)
}

func main() {
	//学习的速度
	learning_rate := 0.1
	n_epoches :=500

	train_N := 6
	test_N := 2
	n_in := 6
	n_out := 2

	train_X := [][]int{
		{1,1,1,0,0,0},
		{1,0,1,0,0,0},
		{1,1,1,0,0,0},
		{0,0,1,1,1,0},
		{0,0,1,1,0,0},
		{0,0,1,1,1,0},
	}

	train_Y := [][]int{
		{1,0},
		{1,0},
		{1,0},
		{0,1},
		{0,1},
		{0,1},

	}

	var classifier LogisticRegresstion
	//构造初始化
	LogisticRegresstion_construct(&classifier, train_N, n_in,n_out)

	for epoch := 0;epoch<n_epoches;epoch++{
		for i:=0;i<train_N;i++{
			//学习的速度
			LogisticRegresstion_train(&classifier, train_X[i], train_Y[i],learning_rate)
		}
	}

	test_X := [][]int{
		{1,0,1,0,0,0},
		{0,0,1,1,1,0},
	}

	test_Y := make([][]float64, test_N)
	for i:=0;i<test_N;i++{
		//开辟数组
		test_Y[i]=make([]float64, n_out)
	}
	for i:=0;i<test_N;i++{
		LogisticRegresstion_predict(&classifier, test_X[i], test_Y[i])
		for j:=0;j<n_out;j++{
			fmt.Printf("%f ,",test_Y[i][j])
		}
		fmt.Println()
	}


}
