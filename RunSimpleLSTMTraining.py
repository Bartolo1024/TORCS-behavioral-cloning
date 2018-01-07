import tensorflow as tf
import numpy as np
import LossFunctions
import TrainInput
from Model import lstm_model
import Normalization as norm
import os

batch_size = 5
time_steps = 15
number_of_sensors = 29
internal_state_size = 20
number_of_efectors = 3

tensorboard_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "summary/")
network_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "saved_network/")

x_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_sensors, time_steps])
y_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_efectors])
target_steering, target_acceleration, target_brake = tf.split(y_placeholder, [1, 1, 1], 1)

#unstack - split input tensor on timesteps
normalized_input = norm.normalize_input(x_placeholder)

input_series = tf.unstack(x_placeholder, axis=2)

outputs_series, cell_state, hidden_state = lstm_model(batch_size, input_series, number_of_sensors, internal_state_size, number_of_efectors)

unnormalized_output = outputs_series[-1]
steering_normalized, acceleration_normalized, brake_normalized, normalized_output = norm.normalize_output(unnormalized_output)

with tf.name_scope("loss"):
    # loss = lf.pow_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param, 12)
    # loss = lf.log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_param)
    loss = LossFunctions.exp_log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized,
                                    target_acceleration, acceleration_normalized, 0.9, batch_size)

with tf.name_scope("optimizer"):
    train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    trainInput = TrainInput.TrainInput(shuffle_data=False, single_race_data_size=10000)

    number_of_epochs = trainInput.max_epochs_index()
    batches_count = trainInput.get_batches_count(batch_size, time_steps)

    for epoch_index in range(number_of_epochs):
        x, y = trainInput.get_chain_train_data(batch_size, number_of_sensors, number_of_efectors, epoch_index)
        _current_cell_state = np.zeros((batch_size, internal_state_size))
        _current_hidden_state = np.zeros((batch_size, internal_state_size))

        print("race index: " + str(epoch_index))

        for batch_index in range(batches_count - time_steps):
            start_index = batch_index
            end_index = start_index + time_steps

            batchX = x[:, :, start_index:end_index]
            batchY = y[:, :, end_index]

            _loss, _train_step, _current_cell_state, _current_hidden_state, _output_series, _normalized_output = sess.run(
                [loss, train_step, cell_state, hidden_state, outputs_series, normalized_output],
                feed_dict={
                    x_placeholder: batchX,
                    y_placeholder: batchY,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                }
            )

            if batch_index%100 == 0:
                print("step", batch_index, "loss", _loss)
                #print(str(_normalized_output) + "_" + str(batchY))

        # save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, network_dirpath + "lstm/model.ckpt")
        print("model saved in file: %s" % save_path)