import tensorflow as tf
data1 = tf.constant(6)
data2 = tf.constant(2)
dataAdd = tf.add(data1,data2)
dataMul = tf.multiply(data1,data2)
dataSub = tf.subtract(data1,data2)
dataDiv = tf.divide(data1,data2)
with tf.Session() as sess:
    print(sess.run(dataAdd))
    print(sess.run(dataMul))
    print(sess.run(dataSub))
    print(sess.run(dataDiv))
print('end!')