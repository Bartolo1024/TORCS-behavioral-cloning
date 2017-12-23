import tensorflow as tf
import TrainInput as ti
import matplotlib.pyplot as plt
import numpy as np
BATCH = 5000
DISPLAY_STEP = 1

#neural network parameters
number_of_features_hidden_a = 37#
number_of_features_hidden_b = 25#
number_of_features_hidden_c = 23#
number_of_features_hidden_d = 15#

number_of_sensors = 27#
number_of_efectors = 3#

x = tf.placeholder(tf.float32, [None, number_of_sensors], name='state')
y_ = tf.placeholder(tf.float32, [None, number_of_efectors], name='action')

#
W1 = tf.Variable(tf.truncated_normal([number_of_sensors, number_of_features_hidden_a], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([number_of_features_hidden_a, number_of_features_hidden_b], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([number_of_features_hidden_b, number_of_features_hidden_c], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([number_of_features_hidden_c, number_of_features_hidden_d], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([number_of_features_hidden_d, number_of_efectors], stddev=0.1))
#
b1 = tf.Variable(tf.zeros(number_of_features_hidden_a))
b2 = tf.Variable(tf.zeros(number_of_features_hidden_b))
b3 = tf.Variable(tf.zeros(number_of_features_hidden_c))
b4 = tf.Variable(tf.zeros(number_of_features_hidden_d))
b5 = tf.Variable(tf.zeros(number_of_efectors))

# visualization
allweights = tf.reshape(W1, [-1])
allbiases = tf.reshape(b1, [-1])

#model
input_layer = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))
hidden_layer_a = tf.nn.relu(tf.add(tf.matmul(input_layer, W2), b2))
hidden_layer_b = tf.nn.relu(tf.add(tf.matmul(hidden_layer_a, W3), b3))
hidden_layer_c = tf.nn.relu(tf.add(tf.matmul(hidden_layer_b, W4), b4))
#y = tf.matmul(tf.nn.tanh(tf.add(tf.matmul(hidden_layer_c, W5), b5)), tf.constant([1.0, 1.0, 1.0, 6.0], shape=(4, 100)))
y = tf.nn.tanh(tf.add(tf.matmul(hidden_layer_c, W5), b5))

cross_entropy = tf.reduce_mean(tf.pow(tf.subtract(y_, y), 2))

train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()

train_accurency = list()
test_accurency = list()

saver = tf.train.Saver()

train_input = ti.TrainInput()
number_of_iterations = int(train_input.get_train_data_count() / BATCH)
print(number_of_iterations)
with tf.Session() as sess:
    sess.run(init)

    for i in range(number_of_iterations):
        batch_x, batch_y = train_input.get_next_batch(batch=BATCH)

        if(i % DISPLAY_STEP == 0):
            acc_trn = sess.run([cross_entropy], feed_dict={x: batch_x, y_: batch_y})

            test_x, test_y = train_input.get_next_test_data(batch=BATCH)
            acc_tst = sess.run([cross_entropy], feed_dict={x: test_x, y_: test_y})

            print("#{} Trn acc={} , Tst acc={} ".format(i, acc_trn, acc_tst))
            train_accurency.append(acc_trn)
            test_accurency.append(acc_tst)

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')

    ax1.plot(train_accurency)
    ax2.plot(test_accurency)

    ax1.set_title("train accurency")
    ax2.set_title("test accurency")

    plt.show()

    sess.close()
    train_input.close()