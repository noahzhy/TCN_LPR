# 基于TCN的端对端车牌识别

## 既存方法

### 传统计算机视觉

![](https://pic1.zhimg.com/80/v2-30af3f6d08836af28a9a69fa75bf823c_1440w.png)

#### 存在的问题

过于依赖字符分割的准确性，字符分割的准确与否直接影响识别的结果。



### CNN-LSTM 结构的基于深度学习的方法



![Sensors | Free Full-Text | Robust Korean License Plate Recognition Based on  Deep Neural Networks | HTML](https://www.mdpi.com/sensors/sensors-21-04140/article_deploy/html/images/sensors-21-04140-g003.png)

#### 存在的问题

LSTM结构较为复杂，在嵌入式设备部署中需要消耗大量的内存。



### 基于TCN 的车牌识别的方法

![](docs\Figure_model.png)



其中 **2D CNN** 提取特征，其中 **CNN Block** 结构如下

$MaxPool(ReLU6(Conv(M))), ReLU6=min(6, max(0,x))$

在 **TCN Block** 中，**Dual Dilated Block** 结构如下

$v_1 = tanh(DilatedConv(M)), tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$

$v_2=ReLU6(DilatedConv(M))$

$Y_1=W[v_1,v_2]\in \R^n$

$Y_2=W(M+Y_1)$

其中 $DilatedConv$ 公式如下

$F(s)=(X*_d f)(s)=\sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}_{s-d \cdot i}$

**d** is dilation scale factor



最后添加CTC loss

$p(Y|X)=\sum_{A\in A_{X,Y}}\prod_{t=1}^{T}p_t(a_t|X)$



最后的结果通过 Greedy Search 对输出进行解码

$A^*=\arg\underset{A}\max\prod{t}=1^Tp_{t}(a_{t}|X)$

