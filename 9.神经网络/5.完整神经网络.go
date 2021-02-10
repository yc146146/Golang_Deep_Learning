package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	"math"
	"math/rand"
	"yinchen.com/文件4/12加强图论/graph"
)

//均方误差
func meanSquareError(y, t *mat.Dense) float64 {
	//
	var a mat.Dense
	//减法
	a.Sub(y, t)
	a.MulElem(&a, &a)
	return 0.5 * mat.Sum(&a)
}

//取得最大值
func argMax(m []float64) int {
	vmax := 0.0
	idx := 0
	for i, v := range m {
		if v > vmax {
			vmax = v
			idx = i
		}
	}
	return idx
}

//交叉熵误差
func crossEntropyError(y, t *mat.Dense) float64 {
	delta := 1e-7
	batchSize, cy := y.Dims()
	rt, ct := t.Dims()
	if batchSize == rt && cy == ct {
		var m []float64
		labels := []float64{}
		for i := 0; i < batchSize; i++ {
			labels = append(labels, float64(argMax(mat.Row(m, i, t))))
		}
		t = mat.NewDense(batchSize, 1, labels)
	}

	sum := 0.0
	for i := 0; i < batchSize; i++ {
		j := int(t.At(i, 0))
		sum += math.Log(y.At(i, j) + delta)
	}
	return -sum / float64(batchSize)
}

func RunError() {
	t := mat.NewDense(1, 10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0})
	y := mat.NewDense(1, 10, []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0})
	fmt.Println(meanSquareError(y, t))
	fmt.Println(crossEntropyError(y, t))

	y = mat.NewDense(1, 10, []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0})
	fmt.Println(meanSquareError(y, t))
	fmt.Println(crossEntropyError(y, t))

}

//假定h  训练0.00001
func NumbericalDiff(f func(float64) float64, x float64) float64 {
	h := 1e-4
	return (f(x+h) - f(x-h)) / (2 * h)
}

//计算训练差别
func functionA(x float64) float64 {
	return 0.01*math.Pow(x, 2) + 0.1*x
}

func functionB(x []float64) float64 {
	return math.Pow(x[0], 2) + math.Pow(x[1], 2)
}

func functionC(x float64) float64 {
	return math.Pow(x, 3) + math.Pow(x, 2)
}

func RunDiff() {
	fmt.Println(NumbericalDiff(functionA, 5))
	fmt.Println(NumbericalDiff(functionA, 2*5))
	fmt.Println(NumbericalDiff(functionA, 3*5))

	fmt.Println(NumbericalDiff(functionC, 5))
	fmt.Println(NumbericalDiff(functionC, 2*5))
	fmt.Println(NumbericalDiff(functionC, 3*5))
}

//矩阵计算向量计算  训练计算差别
func numericalGradient(f func([]float64) float64, x []float64) []float64 {
	h := 1.0e-4
	grad := []float64{}

	for idx := 0; idx < len(x); idx++ {
		tmpVal := x[idx]

		x[idx] = tmpVal + h
		fxh1 := f(x)

		x[idx] = tmpVal - h
		fxh2 := f(x)

		grad = append(grad, (fxh1-fxh2)/(2*h))
		x[idx] = tmpVal

	}

	return grad

}

//矩阵计算向量计算  训练计算差别
func numericalGradient1d(f func(dense *mat.Dense) float64, x *mat.Dense) *mat.Dense {
	h := 1.0e-4
	_, c := x.Dims()
	grad := mat.NewDense(1, c, nil)

	for idx := 0; idx < c; idx++ {
		tmpVal := x.At(1, idx)

		x.Set(1, idx, tmpVal+h)
		fxh1 := f(x)

		x.Set(1, idx, tmpVal-h)
		fxh2 := f(x)

		grad.Set(1, idx, (fxh1-fxh2)/(2*h))
		x.Set(1, idx, tmpVal)

	}

	return grad

}

func numericalGradient2d(f func(dense *mat.Dense) float64, x *mat.Dense) *mat.Dense {

	r, c := x.Dims()
	grad := mat.NewDense(1, c, nil)

	for idx := 0; idx < r; idx++ {
		X := make([]float64, c)
		mat.Row(X, idx, x)
		g := numericalGradient1d(f, mat.NewDense(1, c, X))

		tmp := make([]float64, c)
		mat.Row(tmp, 1, g)
		grad.SetRow(idx, tmp)

	}

	return grad

}

//训练100次, 0.001
func GradienDesent(f func([]float64) float64, xInit []float64, lr float64, stepNum int) []float64 {
	x := make([]float64, cap(xInit))
	copy(x, xInit)

	for i := 0; i < stepNum; i++ {
		grad := numericalGradient(f, x)
		for j := 0; j < len(grad); j++ {
			x[j] -= lr * grad[j]
		}
	}

	return x
}

func RunGrad() {
	fmt.Println(numericalGradient(functionB, []float64{3.0, 4.0, 2.3}))
	fmt.Println(numericalGradient(functionB, []float64{0.0, 2.0, 1.2}))
	fmt.Println(numericalGradient(functionB, []float64{3.0, 2.0, 9.2}))

	fmt.Println(GradienDesent(functionB, []float64{-3.0, 4.0}, 0.1, 100))
	fmt.Println(GradienDesent(functionB, []float64{-3.0, 4.0}, 10, 100))
	fmt.Println(GradienDesent(functionB, []float64{-3.0, 4.0}, 1e-10, 100))

}

//简单神经网络
type SimpleNet struct {
	W *mat.Dense
}

func NewSimpleNet() *SimpleNet {
	r := 2
	c := 3

	seed := []float64{}

	for i := 0; i < r*c; i++ {
		seed = append(seed, rand.Float64())
	}

	return &SimpleNet{mat.NewDense(r, c, seed)}

}

func (n *SimpleNet) Predict(x *mat.Dense) mat.Dense {
	var a mat.Dense
	a.Mul(x, n.W)
	return a
}

func softMax(a *mat.Dense) *mat.Dense {
	y := mat.DenseCopyOf(a)
	c := mat.Max(y)

	y.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v - c)
	}, y)

	sum := mat.Sum(y)
	y.Apply(func(i, j int, v float64) float64 {
		return v / sum
	}, y)

	return y
}

//损失函数
func (n *SimpleNet) loss(x, t *mat.Dense) float64 {
	z := n.Predict(x)
	y := softMax(&z)
	loss := crossEntropyError(y, t)

	return loss
}

func RunNet() {
	//新建神经网络
	net := NewSimpleNet()

	fmt.Println(net.W)

	x := mat.NewDense(1, 2, []float64{0.6, 0.9})
	p := net.Predict(x)
	fmt.Println(p)

	t := mat.NewDense(1, 3, []float64{0, 0, 1})
	loss := net.loss(x, t)

	f := func(dense *mat.Dense) float64 {
		return net.loss(x, t)
	}
	fmt.Println(loss)

	fmt.Println(f)

}

//双层神经网络
type TowlayerNet struct {
	W1, b1, W2, b2 *tensor.Dense
}

//随机数偏差
func gaussian() float64 {
	x := rand.Float64()
	y := rand.Float64()
	return math.Sqrt(-1*math.Log(x)) * math.Cos(2*math.Pi*y)
}

//生成随机向量
func randomDense(dims ...int) *tensor.Dense {
	total := 1
	for _, i := range dims {
		total *= i
	}
	data := []float64{}

	for i := 0; i < total; i++ {
		data = append(data, gaussian())
	}
	return tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))

}

//新建双层神经网络
func NewTowlayerNet(inputsize, hiddensize, outputsize int, weightInitStd float64) *TowlayerNet {
	ret := &TowlayerNet{}

	ret.W1 = randomDense(inputsize, hiddensize)
	ret.W2 = randomDense(hiddensize, outputsize)
	ret.b1 = randomDense(hiddensize)
	ret.b2 = randomDense(outputsize)

	ret.W1, _ = ret.W1.MulScalar(weightInitStd, false)
	ret.b1.Zero()
	ret.W2, _ = ret.W1.MulScalar(weightInitStd, false)
	ret.b2.Zero()

	return ret

}

func sigmoid(a *tensor.Dense) *tensor.Dense {
	ret := a.Clone().(*tensor.Dense)
	it := ret.Iterator()
	for !it.Done() {
		i, _ := it.Next()
		v := ret.Get(i).(float64)
		ret.Set(i, 1/(1+math.Exp(-v)))
	}
	return ret
}

func max(a *tensor.Dense) float64 {
	i, _ := a.Max()
	return i.ScalarValue().(float64)
}

//二维矩阵 取得最大
func softmax(a *tensor.Dense) *tensor.Dense {
	imax, _ := a.Info().Shape().DimSize(0)
	jmax, _ := a.Info().Shape().DimSize(1)
	ret := a.Clone().(*tensor.Dense)
	amax, _ := a.Max(1)
	for i := 0; i < imax; i++ {
		s := 0.0
		for j := 0; j < jmax; j++ {
			aij, _ := ret.At(i, j)
			mi, _ := amax.At(i)
			v := math.Exp(aij.(float64) - mi.(float64))
			s += v

			//计算好设置
			ret.SetAt(v, i, j)
		}

		for j := 0; j < jmax; j++ {
			aij, _ := ret.At(i, j)
			ret.SetAt(aij.(float64)/s, i, j)
		}

	}
	return ret
}

func addVector(dense, vec *tensor.Dense) {
	imax, _ := dense.Info().Shape().DimSize(0)
	jmax, _ := dense.Info().Shape().DimSize(1)

	for i := 0; i < imax; i++ {
		for j := 0; j < jmax; j++ {

			aij, _ := dense.At(i, j)
			vj, _ := vec.At(j)
			dense.SetAt(aij.(float64)+vj.(float64), i, j)

		}
	}

}

func (this *TowlayerNet) predict(x *tensor.Dense) *tensor.Dense {
	a1, err := x.TensorMul(this.W1, []int{1}, []int{0})
	if err != nil{
		fmt.Println(err)
	}
	addVector(a1, this.b1)
	z1 := sigmoid(a1)

	a2, err := x.TensorMul(this.W2, []int{1}, []int{0})

	if err != nil{
		fmt.Println(err,z1,a2)
	}

	addVector(a2, this.b2)
	//取得最大概率
	y := softmax(a2)

	return y

}



func (this *TowlayerNet) loss(x,t *tensor.Dense) float64 {
	y := this.predict(x)
	return crossEntryErrorTensor(y,t)
}

//交叉熵
func crossEntryErrorTensor(y, t *tensor.Dense) float64 {
	delta := 1e-7

	//形状
	yShape := y.Info().Shape()

	batchSize,_ := yShape.DimSize(0)
	cy,_ := yShape.DimSize(1)
	tShape := t.Info().Shape()
	rt,_:=tShape.DimSize(0)
	ct,_ := tShape.DimSize(1)
	if batchSize == rt && cy == ct{
		labels := []float64{}
		argmaxs,  _ := t.Argmax(1)
		for i:=0;i<batchSize;i++{
			a,_ := argmaxs.At(i)
			f := float64(a.(int))
			//叠加
			labels = append(labels,f)
		}
		//t = tensor.New(tensor.WithShape(batchSize,1),tensor.WithBacking(labels))
	}
	sum := 0.0

	for i:=0;i<batchSize;i++{
		j,_ := t.At(i,0)
		yij,_:= y.At(i,int(j.(float64)))
		sum += math.Log(yij.(float64)+delta)
	}
	return -sum/float64(batchSize)
}

//func (t * tensor.Dense)Error()error{
//
//}

//精确度计算
func (this *TowlayerNet) accuracy(x,t *tensor.Dense) float64 {
	y := this.predict(x)
	ya, _ := y.Argmax(1)
	ta, _ := t.Argmax(1)


	sum := 0
	size := 0
	it := ya.Iterator()
	for !it.Done(){
		i,_ := it.Next()
		yi := ya.Get(i).(float64)
		ti := ta.Get(i).(float64)

		if yi == ti{
			sum++
		}
		size ++
	}

	return float64(sum)/float64(size)


}

func numericalGradientTensor(f func(dense *tensor.Dense) float64, x *tensor.Dense) *tensor.Dense {
	h := 1e-4
	grad := x.Clone().(*tensor.Dense)
	it := x.Iterator()
	for !it.Done(){
		i,_ := it.Next()
		tmpval := x.Get(i).(float64)
		x.Set(i, tmpval+h)
		fxh1 := f(x)
		x.Set(i, tmpval-h)
		fxh2 := f(x)

		grad.Set(i,(fxh1*fxh2)/(2*h))
	}

	return grad
}

//数值计算
func (this *TowlayerNet) numericalGradient(t, x *tensor.Dense) *TowlayerNet {
	lossw := func(_ *tensor.Dense)float64{
		return this.loss(x,t)
	}

	grads := &TowlayerNet{}
	grads.W1 = numericalGradientTensor(lossw, this.W1)
	grads.b1 = numericalGradientTensor(lossw, this.b1)
	grads.W2 = numericalGradientTensor(lossw, this.W2)
	grads.b2 = numericalGradientTensor(lossw, this.b2)
	return grads
}

func (this *TowlayerNet) Add(network *TowlayerNet) {
	this.W1, _ = this.W1.Add(network.W1)
	this.b1, _ = this.b1.Add(network.b1)
	this.W2, _ = this.W2.Add(network.W2)
	this.b2, _ = this.b2.Add(network.b2)
}

func (this *TowlayerNet) AMulScalar(n float64) {
	this.W1, _ = this.W1.MulScalar(n,false)
	this.b1, _ = this.b1.MulScalar(n,false)
	this.W2, _ = this.W2.MulScalar(n,false)
	this.b2, _ = this.b2.MulScalar(n,false)
}

//随机选取
func RandomBatchMask(limit, size int) []int {

	store := map[int]int{}
	for len(store)<size{
		i:= rand.Intn(limit)
		store[i]=i
	}

	ret := []int{}
	for i,_ := range store{
		ret = append(ret, i)
	}



	return ret
}

func byte2float63(bs []byte) []float64 {

	ret := []float64{}

	for _,b := range bs{
		ret = append(ret , float64(b))
	}

	return ret
}

func RunTowLayerNetWork(){
	t1 := tensor.New(tensor.WithShape(2,3), tensor.WithBacking(tensor.Range(tensor.Float64,0,6)))
	t2 := tensor.New(tensor.WithShape(3,4), tensor.WithBacking(tensor.Range(tensor.Float64,0,12)))
	t3,_ := t1.TensorMul(t2, []int{1},[]int{0})
	fmt.Println(t1)
	fmt.Println(t2)
	fmt.Println(t3)
}

func RunMinst(){

}

func main() {

	//RunError()

	//RunDiff()

	//RunGrad()

	//RunNet()

	RunTowLayerNetWork()

}
