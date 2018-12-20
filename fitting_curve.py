import tensorflow as tf
import numpy as np

# 原始数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.3

# 损失函数
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weights * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))

# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))