#1 import 2 string 3 print
import cv2
import tensorflow as tf
print('hello opencv')
hello = tf.constant('hello tf!')
sess = tf.Session()
print(sess.run(hello).decode())
#常量 sess print