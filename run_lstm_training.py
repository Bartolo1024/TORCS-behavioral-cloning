import tensorflow as tf
import numpy as np
import loss_functions
import train_input
from model import lstm_model
import normalization as norm
import os
import shared_preferences

batch_size = shared_preferences.batch_size
time_steps = shared_preferences.backpropagation_size
number_of_sensors = shared_preferences.number_of_sensors
internal_state_size = shared_preferences.internal_state_size
number_of_efectors = shared_preferences.number_of_efectors
number_of_lstm_layers = shared_preferences.number_of_lstm_layers

tensorboard_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "summary/")
network_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "saved_network/lstm/")

def run(batch, lambda_param = 0.9, learning_rate = 0.005, namespace=""):
    tf.reset_default_graph()
    x_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_sensors, time_steps])
    y_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_efectors])
    target_steering, target_acceleration = tf.split(y_placeholder, [1, 1], 1)

    #unstack - split input tensor on timesteps
    normalized_input = norm.normalize_input(x_placeholder)

    input_series = tf.unstack(normalized_input, axis=2)

    outputs_series, current_state, init_state = lstm_model(batch_size,
                                                           input_series,
                                                           number_of_sensors,
                                                           internal_state_size,
                                                           number_of_efectors,
                                                           number_of_lstm_layers)

    unnormalized_output = outputs_series[-1]
    steering_normalized, acceleration_normalized, normalized_output = norm.normalize_output(unnormalized_output)

    #loss for all outputs
    loss = 0

    with tf.name_scope("loss"):
        # losses = []
        # for output in outputs_series:
        #     steering_normalized, acceleration_normalized, normalized_output = norm.normalize_output(output)
        #     losses.append(loss_functions.exp_log_loss_function(target_steering, steering_normalized, target_acceleration,
        #                                                 acceleration_normalized, lambda_param))
        # loss = tf.reduce_mean(losses)


        # loss = lf.pow_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param, 12)
        # loss = lf.log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param)
        loss = loss_functions.exp_log_loss_function(target_steering, steering_normalized, target_acceleration,
                                                    acceleration_normalized, lambda_param)
        tf.summary.scalar(
            "lr: " + "{:.7f}".format(learning_rate) + "lambda: " + "{:.7f}".format(lambda_param) + "batch: " + str(
                batch), loss)

    with tf.name_scope("optimizer"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        trainInput = train_input.TrainInput(shuffle_data=False, single_race_data_size=10000)

        number_of_epochs = 100
        batches_count = trainInput.get_batches_count(batch_size)
        writer = tf.summary.FileWriter(tensorboard_dirpath + "train", sess.graph)

        for epoch_index in range(number_of_epochs):
            x, y = trainInput.get_chain_train_data(batch_size, number_of_sensors)
            _current_state = np.zeros((number_of_lstm_layers, 2, batch_size, internal_state_size))

            for batch_index in range(batches_count - time_steps + 1):
                start_index = batch_index
                end_index = start_index + time_steps

                batchX = x[:, :, start_index:end_index]
                batchY = y[:, :, end_index - 1]

                _loss, _train_step, _current_state, _output_series, summary = sess.run(
                    [loss, train_step, current_state, outputs_series, merged],
                    feed_dict={
                        x_placeholder: batchX,
                        y_placeholder: batchY,
                        init_state: _current_state
                    }
                )



                if batch_index % 50 == 0:
                    print("race: " + str(number_of_epochs + epoch_index) + " loss " + str(_loss))

                writer.add_summary(summary, global_step=batch_index + epoch_index * batches_count)

        # save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, network_dirpath + "model.ckpt")
        print("model saved in file: %s" % save_path)

if tf.gfile.Exists(tensorboard_dirpath):
    tf.gfile.DeleteRecursively(tensorboard_dirpath)

run(batch_size, lambda_param=0.9, learning_rate=0.00005)