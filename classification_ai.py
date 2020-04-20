import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import numpy as np

#[털, 날개] -> [기타, 포유류, 조류]
x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])

#기타 = 0, 포유류 = 1, 조류 = 2
# 기타 = [1,0,0]
# 포유류 = [0,1,0]
# 조류 = [0,0,1]

y_data = np.array([
    [1,0,0], #기타
    [0,1,0], #포유류
    [0,0,1], #조류
    [1,0,0],
    [1,0,0],
    [0,0,1] 
    ])

#[0,0] -> [1,0,0] 기타
#[1,0] -> [0,1,0] 포유류
#[0,1] -> [0,0,1] 조류

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

#출력값 다듬기
model = tf.nn.softmax(L)

#손실 함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

#학습
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train_op, feed_dict = {X: x_data, Y: y_data})

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)

print("predict: ", sess.run(prediction, feed_dict={X: x_data}))
print("real_num: ",sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('correctness: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

#sess.close()

