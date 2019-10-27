# Batch normlization(批归一化)

### 为什么需要Batch normlization(批归一化)

当神经网络训练时，除输入层外的每层的输入数据分布是一直在变化的，随着网络逐渐加深，数据分布发生变动，导致训练收敛缓慢，因此需要我们去将每层神经元的输入数据拉回一个标准的正态分布，让梯度一直能保持在一个较大的状态，使神经网络对参数的更新效率更高。Batch normlization是一种最新的对数据差异进行处理的手段。通过对数据进行规范化处理，使得输出的均值为0，方差为1。

### 如何实现Batch normlization(批归一化)

![]( https://github.com/Shigurea/boke/blob/master/pic/1093303-20180219084749642-1647361064.png )

### 神经网络使用BN的位置

![]( https://github.com/Shigurea/boke/blob/master/pic/1192699-20180405213859690-1933561230.png )

如图这是一个深层神经网络其中的两层结构，对神经网络做BN的位置就在X=WU+B激活值获得之后，非线性函数（激活函数）变换之前，其图示如下： 

![]( https://github.com/Shigurea/boke/blob/master/pic/1192699-20180405213955224-1791925244.png )

### Tensorflow实现BN

tf.layers.batch_normalization(input, training)

在训练时training = True，在测试时traning = False