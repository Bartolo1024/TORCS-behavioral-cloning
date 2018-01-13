#!/usr/bin/env python

import random
import os.path
from warnings import _add_filter
import tensorflow as tf
import train_input as ti
import loss_functions as lf
import normalization
import model
import shared_preferences

MIN_BATCH = 100
MAX_BATCH = 10000
DISPLAY_STEP = 10
tensorboard_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "summary/")
network_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "saved_network/")
print(tensorboard_dirpath)

number_of_sensors = shared_preferences.number_of_sensors
number_of_efectors = shared_preferences.number_of_efectors
layers_shapes = shared_preferences.layers_shapes

def run(batch, lambda_param = 0.9, learning_rate = 0.005, namespace=""):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, number_of_sensors], name='state')
    y_ = tf.placeholder(tf.float32, [None, number_of_efectors], name='action')
    target_steering, target_acceleration = tf.split(y_, [1, 1], 1)

    # model and normalization
    normalized_input = normalization.normalize_input(x)
    unnormalized_output = model.mlp_model(normalized_input, number_of_sensors, number_of_efectors, layers_shapes)
    steering_normalized, acceleration_normalized, y = normalization.normalize_output(unnormalized_output)

    with tf.name_scope("loss"):
        #loss = lf.pow_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param, 12)
        #loss = lf.log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param)
        loss = lf.exp_log_loss_function(target_steering, steering_normalized, target_acceleration, acceleration_normalized, lambda_param, batch)
        tf.summary.scalar("lr: " + "{:.7f}".format(learning_rate) + "lambda: " + "{:.7f}".format(lambda_param) + "batch: " + str(batch), loss)

    with tf.name_scope(namespace):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    train_input = ti.TrainInput(acceleration_brake_merged = True)
    number_of_iterations = int(train_input.get_train_data_count() / batch) * 1000

    with tf.Session() as sess:

        sess.run(init)

        writter = tf.summary.FileWriter(tensorboard_dirpath + "train", sess.graph)
        test_writer = tf.summary.FileWriter(tensorboard_dirpath + "test")

        for i in range(number_of_iterations):
            batch_x, batch_y = train_input.get_next_batch(batch=batch)
            _, traning_loss, summary, output = sess.run([train_step, loss, merged, y], feed_dict={x: batch_x, y_: batch_y})

            writter.add_summary(summary, global_step=i)

            test_x, test_y = train_input.get_next_test_data(batch=batch)
            test_loss, summary = sess.run([loss, merged], feed_dict={x: test_x, y_: test_y})

            test_writer.add_summary(summary, global_step=i)

            if (i % DISPLAY_STEP == 0):
                print("#{} Trn loss={} , Tst loss={} ".format(i, traning_loss, test_loss))
                print(output[0])
                print(batch_y[0])

        # save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, network_dirpath + "model.ckpt")
        print("model saved in file: %s" % save_path)

        writter.close()
        test_writer.close()

        sess.close()
        train_input.close()

# for i in range(0, 10):
#     lr_sub = random.random() * 4
#     learning_rate = pow(0.1, (3 + lr_sub))
#     for j in range(0, 10):
#         batch = random.randrange(MIN_BATCH, MAX_BATCH, 10)
#         for k in range(0, 10):
#             lambda_param = random.random()
#             print("batch: " + str(batch) + "learning rate: " + str(learning_rate) + "lambda_param:" + str(lambda_param))
#             run(batch, lambda_param, learning_rate)

if tf.gfile.Exists(tensorboard_dirpath):
    tf.gfile.DeleteRecursively(tensorboard_dirpath)

run(1000, 0.9, 0.0005) #best for exponent

