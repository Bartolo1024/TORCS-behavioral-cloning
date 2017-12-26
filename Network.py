import tensorflow as tf
import TrainInput as ti
import matplotlib.pyplot as plt
import numpy as np
BATCH = 5000
DISPLAY_STEP = 1

#neural network parameters
number_of_features_hidden_a = 35#
number_of_features_hidden_b = 40#
number_of_features_hidden_c = 40#
number_of_features_hidden_d = 27#
number_of_features_hidden_e = 35#

number_of_sensors = 27#
number_of_efectors = 3#

x = tf.placeholder(tf.float32, [None, number_of_sensors], name='state')
y_ = tf.placeholder(tf.float32, [None, number_of_efectors], name='action')

#
W1 = tf.Variable(tf.truncated_normal([number_of_sensors, number_of_features_hidden_a], stddev=0.1), name="W1")
W2 = tf.Variable(tf.truncated_normal([number_of_features_hidden_a, number_of_features_hidden_b], stddev=0.1), name="W2")
W3 = tf.Variable(tf.truncated_normal([number_of_features_hidden_b, number_of_features_hidden_c], stddev=0.1), name="W3")
W4 = tf.Variable(tf.truncated_normal([number_of_features_hidden_c, number_of_features_hidden_d], stddev=0.1), name="W4")
W5 = tf.Variable(tf.truncated_normal([number_of_features_hidden_d, number_of_features_hidden_e], stddev=0.1), name="W5")
W6 = tf.Variable(tf.truncated_normal([number_of_features_hidden_e, number_of_efectors], stddev=0.1), name="W6")
#
b1 = tf.Variable(tf.zeros(number_of_features_hidden_a), name="b1")
b2 = tf.Variable(tf.zeros(number_of_features_hidden_b), name="b2")
b3 = tf.Variable(tf.zeros(number_of_features_hidden_c), name="b3")
b4 = tf.Variable(tf.zeros(number_of_features_hidden_d), name="b4")
b5 = tf.Variable(tf.zeros(number_of_features_hidden_e), name="b5")
b6 = tf.Variable(tf.zeros(number_of_efectors), name="b6")

# visualization
allweights = tf.reshape(W1, [-1])
allbiases = tf.reshape(b1, [-1])

#model

#normalization
speedX, speedY, speedZ, rpm, wheelSpinVel, track = tf.split(x, [1, 1, 1, 1, 4, 19], 1)
normalized_speed_x = tf.divide(speedX, 500)
normalized_speed_y = tf.divide(speedY, 500)
normalized_speed_z = tf.divide(speedZ, 500)
rpm_normalized = tf.divide(rpm, 10000)
wheel_spin_velocity_normalized = tf.divide(wheelSpinVel, 1000)
track_normalized = track
normalized_input = tf.concat([normalized_speed_x, normalized_speed_y, normalized_speed_z, rpm_normalized, wheel_spin_velocity_normalized, track], 1)
input_layer = tf.nn.tanh(tf.add(tf.matmul(normalized_input, W1), b1))

hidden_layer_a = tf.nn.relu(tf.add(tf.matmul(input_layer, W2), b2))
hidden_layer_b = tf.nn.relu(tf.add(tf.matmul(hidden_layer_a, W3), b3))
hidden_layer_c = tf.nn.relu(tf.add(tf.matmul(hidden_layer_b, W4), b4))
hidden_layer_d = tf.nn.relu(tf.add(tf.matmul(hidden_layer_c, W5), b5))

unnormalized_output = tf.add(tf.matmul(hidden_layer_d, W6), b6)
steering, acceleration, brake = tf.split(unnormalized_output, [1, 1, 1], 1)
steering_normalized = tf.tanh(steering)             #steering values e(-1, 1)
acceleration_normalized = tf.sigmoid(acceleration)  #acceleration values e(0, 1)
brake_normalized = tf.sigmoid(brake)                #acceleration values e(0, 1)

y = tf.concat([steering_normalized, acceleration_normalized, brake_normalized], 1)
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(y_, y), 2)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

train_accurency = list()
test_accurency = list()

saver = tf.train.Saver()

train_input = ti.TrainInput()
number_of_iterations = int(train_input.get_train_data_count() / BATCH)
with tf.Session() as sess:
    sess.run(init)

    for i in range(number_of_iterations):
        batch_x, batch_y = train_input.get_next_batch(batch=BATCH)

        if(i % DISPLAY_STEP == 0):
            acc_trn, output = sess.run([cross_entropy, y], feed_dict={x: batch_x, y_: batch_y})

            test_x, test_y = train_input.get_next_test_data(batch=BATCH)
            acc_tst = sess.run([cross_entropy], feed_dict={x: test_x, y_: test_y})

            print("#{} Trn acc={} , Tst acc={} ".format(i, acc_trn, acc_tst))
            train_accurency.append(acc_trn)
            test_accurency.append(acc_tst)

            print(output[0])
            print(batch_y[0])

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')

    ax1.plot(train_accurency)
    ax2.plot(test_accurency)

    ax1.set_title("train loss")
    ax2.set_title("test loss")

    plt.show()

    #save model
    save_path = saver.save(sess, "/home/bartolo/Projects/Torcs/TrainedNeuralNetwork/model.ckpt")
    print("model saved in file: %s" % save_path)

    sess.close()
    train_input.close()