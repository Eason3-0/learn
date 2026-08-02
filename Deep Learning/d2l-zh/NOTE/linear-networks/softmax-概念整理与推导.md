## softmax函数
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) 
$$
$$
\hat{y}_j = \frac{\exp{(o_j)}}{\sum_k \exp{o_k}}
$$
假设我们读取了一个批量的样本$\mathbf{X}$，其中特征维度（输入数量）为$d$，批量大小为$n$。此外，假设我们在输出中有$q$个类别。那么小批量样本的特征为$\mathbf{X} \in \mathbb{R}^{n \times d}$，权重为$\mathbf{W} \in \mathbb{R}^{d \times q}$，偏置为$\mathbf{b} \in \mathbb{R}^{1\times q}$。

softmax回归的矢量计算表达式为：
$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$

## 损失函数
接下来，我们需要一个损失函数来度量预测的效果。我们将使用最大似然估计，这与在线性回归中的方法相同。

### 对数似然
softmax函数给出了一个向量$\hat{\mathbf{y}}$，我们可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。例如，$\hat{y}_1$=$P(y=\text{猫} \mid \mathbf{x})$。假设整个数据集$\{\mathbf{X}, \mathbf{Y}\}$具有$n$个样本，其中索引$i$的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成。
我们可以将估计值与实际值进行比较：

  

$$

P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).

$$
根据最大似然估计，我们最大化$P(\mathbf{Y} \mid \mathbf{X})$，相当于最小化负对数似然：
$$

-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
=\sum_{i=1}^{n} -\log\prod_{j=1}^{q}(\hat{y}_j^{(i)})^{y_j^{(i)}}
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),

$$
其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，定义损失函数（交叉熵损失）为：

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$

## softmax及其导数
由于softmax和相关的损失函数很常见，因此我们需要更好地理解它的计算方式。利用softmax的定义，我们得到：

  

$$

\begin{aligned}

l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\

&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\

&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.

\end{aligned}

$$
考虑相对于任何未规范化的预测$o_j$的导数，我们得到：

$$

\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.

$$
