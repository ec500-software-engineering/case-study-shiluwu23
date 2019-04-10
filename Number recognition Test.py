import tensorflow as tf
import input_data

# download MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# weights，biases，initial=0
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

# construct model
x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x, weights) + biases)                                   # predicted value
y_real = tf.placeholder("float", [None, 10])                                        # real value

cross_entropy = -tf.reduce_sum(y_real * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# train
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                                # ramdomly select 100 data to train（Stochastic Gradient Descent，SGD）”
    sess.run(train_step, feed_dict={x: batch_xs, y_real:batch_ys})                  # do train_step

    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_real, 1))       # compare y and y_real
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             # compute accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))