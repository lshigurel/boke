## BatchNormalization的错误使用

```python
input1 = tf.ones[1,4,4,3]
output1 = tf.layers.batch_normlization(cov1,training = True)
loss = ...		#损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minize(loss)

with tf.Session() as sess:
    sess.run(optimizer)

```

在训练神经网络时，把tf.layers.batch_normlization中的traning参数设置为True，在使用模型时讲training参数设置为False。训练时误差能下降得很低，但是在使用模型时，正确率却非常低，导致这个错误出现的原因就是没用调用update_ops，以下是正确使用的代码

```python
input1 = tf.ones[1,4,4,3]
output1 = tf.layers.batch_normlization(cov1,training = True)
loss = ...		#损失函数
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minize(loss)

with tf.Session() as sess:
    sess.run(optimizer)
```

两段代码差别主要在

```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
```

首先说一下tf.control_dependencies(control_inputs)

此函数指定某些操作执行的依赖关系，返回一个控制依赖的上下文管理器， 使用 **with** 关键字可以让在这个上下文环境中的操作都在 **control_inputs** 执行 。

```python
with tf.control_dependencies([a, b]):
2     c = ....
3     d = ...
```

 在执行完 a，b 操作之后，才能执行 c，d 操作。意思就是 c，d 操作依赖 a，b 操作 。

然后是tf.GraphKeys.UPDATE_OPS

 这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合函数tf.control_dependencies使用，在BN中，即为更新均值(mean)和方差(variance)的操作。我们需要保证这两个操作在optimizer之前完成， 所以要通过第一个函数来实现控制依赖，第二个函数来让操作在更新参数后进行。