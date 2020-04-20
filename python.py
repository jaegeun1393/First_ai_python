import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
#sess = tf.Session()

x_data = [1,2,3]
y_data = [1,2,3]

#-1.0 부터 1 까지 random 으로 변수를 만들어낸다. (초기화 한다)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#place holder 에 이름을 부여한다.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

#X, Y 관계 만들기
hyp = W * X + b

#손실 함수 작성 계산한 값과 실제값이 얼마나 차이가 나는가 를 나타낸다.
cost = tf.reduce_mean(tf.square(hyp - Y))

#손실함수가 적은 연산 그래프를 생성한다.
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost)

#변수 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

#학습 for 을 이용하여 100번을 수행하고 손실값을 출력한다
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict = {X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n===Test===")

    print("X: 5, y:", sess.run(hyp, feed_dict={X: 5}))
    print("X: 2.5, y:", sess.run(hyp, feed_dict={X: 2.5}))

sess.close()

