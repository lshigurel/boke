# AlexNet

## 结构

![]( https://github.com/Shigurea/boke/blob/master/pic/20190614111847701.png )

如上图所示，AlexNet采用两台GPU，因此有两个流程图

卷积层1：卷积核大小为11x11x3，数量为48个，两台GPU共96个卷积核，步长为4，采用ReLU激活函数 ，池化层大小为3x3，步长为2，采用LRN归一化。

卷积层2：像素层扩展两个像素，卷积核大小为5x5x48，数量为128个，两台GPU共256个卷积核，步长为1，采用ReLU激活函数，池化层大小为3x3，步长为2，采用LRN归一化。

卷积层3：像素层扩展一个像素，卷积核大小为3x3x128，数量为192个，两台GPU共384个卷积核，步长为1，采用ReLU激活函数，无池化层。

卷积层4：像素层扩展一个像素，卷积核大小为3x3x192，数量为192个，

两台GPU共384个卷积核，步长为1，采用ReLU激活函数，无池化层。

卷积层5：像素层扩展一个像素，卷积核大小为3x3x192，数量为256个，两台GPU共256个卷积核，步长为1，采用ReLU激活函数，池化层大小为3x3，步长为2。

全连接层6：2048个神经元，两台GPU共4096个神经元，使用Dropout防止过拟合，输出4096x1的向量。

全连接层7：2048个神经元，两台GPU共4096个神经元，使用Dropout防止过拟合，输出4096x1的向量。

全连接层8：1000个神经元。

## 优点

1、训练时使用Dropout随机忽略一部分神经元，来避免过拟合。

2、 成功使用ReLU作为CNN的激活函数，在较深的网络中效果超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题 。

3、 使用最大池化，避免了平均池化的模糊效果 。

4、 提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应较大的值变得相对更大，并抑制其他较小的神经元，增强了模型的泛化能力 。

5、 基于GPU使用CUDA加速神经网络的训练 。

## 代码实现

```python
n_classes = 2 
n_fc1 = 4096
n_fc2 = 2048
#构建模型
x = tf.placeholder(tf.float32, [None,227,227,3])
y = tf.placeholder(tf.float32, [None,n_classes])

W_conv = {     'conv1':tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.0001)),     'conv2':tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.01)),     'conv3':tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.01)),     'conv4':tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.01)),     'conv5':tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.01)),     'fc1':tf.Variable(tf.truncated_normal([6*6*256,n_fc1],stddev=0.1)),     'fc2':tf.Variable(tf.truncated_normal([n_fc1,n_fc2],stddev=0.1)),     'fc3':tf.Variable(tf.truncated_normal([n_fc2,n_classes],stddev=0.1)) }

b_conv = {     'conv1':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[96])),     'conv2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),     'conv3':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),     'conv4':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),     'conv5':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),     'fc1':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc1])),     'fc2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc2])),     'fc3':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[n_classes])) }

x_image = tf.reshape(x,[-1,227,227,3])
#卷积层1 
conv1 = tf.nn.conv2d(x_image ,W_conv['conv1'],strides=[1,4,4,1],padding='VALID') 
conv1 = tf.nn.bias_add(conv1,b_conv['conv1']) 
conv1 = tf.nn.relu(conv1) 
#池化层1 
pool1 = tf.nn.avg_pool2d(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID') 
#LRN
norm1 = tf.nn.lrn(pool1,5,bias=1.0,alpha=0.001/9.0,beta=0.75) 

#卷积层2 
conv2 = tf.nn.conv2d(norm1,W_conv['conv2'],strides=[1,1,1,1],padding='SAME') 
conv2 = tf.nn.bias_add(conv2,b_conv['conv2']) 
conv2 = tf.nn.relu(conv2) 
#池化层2 
pool2 = tf.nn.avg_pool2d(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID') 
#LRN 
norm2 = tf.nn.lrn(pool2,5,bias=1.0,alpha=0.001/9.0,beta=0.75)

#卷积层3 
conv3 = tf.nn.conv2d(norm2,W_conv['conv3'],strides=[1,1,1,1],padding='SAME') 
conv3 = tf.nn.bias_add(conv3,b_conv['conv3']) 
conv3 = tf.nn.relu(conv3)  

#卷积层4 
conv4 = tf.nn.conv2d(conv3,W_conv['conv4'],strides=[1,1,1,1],padding='SAME') 
conv4 = tf.nn.bias_add(conv4,b_conv['conv4'])
conv4 = tf.nn.relu(conv4)

#卷积层5 
conv5 = tf.nn.conv2d(conv4,W_conv['conv5'],strides=[1,1,1,1],padding='SAME') 
conv5 = tf.nn.bias_add(conv5,b_conv['conv5']) 
conv5 = tf.nn.relu(conv5) 
#池化层3 
pool3 = tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID') 

reshape = tf.reshape(pool3,[-1,6*6*256]) 
#全连接层1 
fc1 = tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1']) fc1 = tf.nn.relu(fc1) fc1 = tf.nn.dropout(fc1,0.5) 
#全连接层2 
fc2 = tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2']) fc2 = tf.nn.relu(fc2) fc2 = tf.nn.dropout(fc2,0.5) 
#全连接层3，分类层 
fc3 = tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3']) print(fc3) 
#定义损失 
loss_funciton = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y)) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_funciton) 
#评估模型 
correct_pred = tf.equal(tf.argmax(fc3,1),tf.argmax(y,1)) accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
```

### 