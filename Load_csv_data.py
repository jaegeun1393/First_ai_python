import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

#load data
data = np.loadtxt('./data.csv', delimiter=',',unpack=True, dtype='float32')

#1열 과 2열은x_data && 3열 부터 나머지는 y_data 에
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

#저장
global_step = tf.Variable(0, trainable=False, name='global_step')

#2중 신경망 구축
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0))
model = tf.matmul(L2, W3)

#손실값 계산
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

#학습, 최적화 할때마다 global_step 이 1씩 증가한
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost, global_step=global_step)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

#saver load 해오는 함수
ckpt = tf.train.get_checkpoint_state('.model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

#global_step 으로 몇번째 인지 구한다

#학습 2번 시킴
for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y:y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

#체크포인트 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

#결과 값 print

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
