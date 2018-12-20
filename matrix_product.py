import tensorflow as tf

matrix1 = tf.constant([[1, 2]])
matrix2 = tf.constant([[3], [4]])

product = tf.matmul(matrix1, matrix2)

# 启动方法1
sess = tf.Session()
res = sess.run(product)
print(res)
sess.close()

# 启动方法2
with tf.Session() as sess:
    res2 = sess.run(product)
    print(res2)