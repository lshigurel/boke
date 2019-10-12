## Tensor张量

﻿在Tensorflow中所有的数据都通过张量的形式来表示。

创建流图（或计算图）就是建立计算模型，执行对话才能提供数据并获得结果。
张量可以简单理解为多维数组。
张量并没有真正保存数字，他保存的是计算过程。

### 张量的属性

Tensor("Add:0", shape=(), dtype=float32)

#### 名字(name):

​      “node:src_ output” : node 节点名称，src_ output 来自节点的第几个输出

#### 形状(shape):

​      张量的维度信息，  shape=() 表示是标量

#### 类型(type):

​      每一个张量会有一个唯一的类型
​     TensorFlow会对参与运算的所有张量进行类型的检查，发现类型不匹配时会报错


    TensorFlow支持14种不同的类型
    
      实数tf. float32, tf.float64
    
      整数tf. int8, tf.int16, tf.int32, tf.int64, tf.uint8
    
      布尓tf.bool ;
    
      复数数tf.complex64, tf.complex128
    
      默认类型:
    
      不带小数点的数会被默认为int32
      带小数点的会被默认为float32

当默认的会话被指定之后可以通过tf.Tensor.eval函数来计算一个张量的取值。

### 变量Variable

#### 创建语句:

​	name_variable = tf.Variable(value, name)

#### 个别变量初始化:

​	init_op = name_variable.initializer()
​	sess.run(init_op)

#### 所有变量初始化:

​	init_op = tf.global_variables_initializer()
​	sess.run(init_op)

#### 人工更新变量语句：

​	update_op = tf.assign(variable_to_be_updated,new_value)

#### 占位符placeholder

​	tf.placeholder(dtype,shape=None,name=None)
​	用Feed提交数据，Fetch提取数据

## Tensorflow运算符


### 算术操作符：+ - * / % 
tf.add(x, y, name=None)        # 加法(支持 broadcasting)
tf.subtract(x, y, name=None)   # 减法
tf.multiply(x, y, name=None)   # 乘法
tf.divide(x, y, name=None)     # 浮点除法, 返回浮点数(python3 除法)
tf.mod(x, y, name=None)        # 取余


### 幂指对数操作符：^ ^2 ^0.5 e^ ln 
tf.pow(x, y, name=None)        # 幂次方
tf.square(x, name=None)        # 平方
tf.sqrt(x, name=None)          # 开根号，必须传入浮点数或复数
tf.exp(x, name=None)           # 计算 e 的次方
tf.log(x, name=None)           # 以 e 为底，必须传入浮点数或复数


### 取符号、负、倒数、绝对值、近似、两数中较大/小的
tf.negative(x, name=None)      # 取负(y = -x).
tf.sign(x, name=None)          # 返回 x 的符号
tf.reciprocal(x, name=None)    # 取倒数
tf.abs(x, name=None)           # 求绝对值
tf.round(x, name=None)         # 四舍五入
tf.ceil(x, name=None)          # 向上取整
tf.floor(x, name=None)         # 向下取整
tf.rint(x, name=None)          # 取最接近的整数 
tf.maximum(x, y, name=None)    # 返回两tensor中的最大值 (x > y ? x : y)
tf.minimum(x, y, name=None)    # 返回两tensor中的最小值 (x < y ? x : y)


### 三角函数和反三角函数
tf.cos(x, name=None)    
tf.sin(x, name=None)    
tf.tan(x, name=None)    
tf.acos(x, name=None)
tf.asin(x, name=None)
tf.atan(x, name=None)   