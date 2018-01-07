import tensorflow as tf
import numpy as np
import LossFunctions
import TrainInput

batch_size = 5
backpropagation_lenght = 15

number_of_sensors = 29
internal_state_size = 20
number_of_efectors = 3

x_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_sensors, backpropagation_lenght])
y_placeholder = tf.placeholder(tf.float32, [batch_size, number_of_efectors, backpropagation_lenght])

init_state = tf.placeholder(tf.float32, [batch_size, internal_state_size])

W_in = tf.get_variable("W1", [number_of_sensors + internal_state_size, internal_state_size], initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
b_in = tf.get_variable("b1", [1, internal_state_size], initializer = tf.zeros_initializer(), dtype=tf.float32)

W_out = tf.get_variable("W2", [internal_state_size, number_of_efectors], initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
b_out = tf.get_variable("b2", [1, number_of_efectors], initializer = tf.zeros_initializer(), dtype=tf.float32)

#unstack - split input tensor on timesteps
input_series = tf.unstack(x_placeholder, axis=2)
output_series = tf.unstack(y_placeholder, axis=2)

current_state = init_state
state_series = []

#forward passes
cell = tf.nn.rnn_cell.BasicRNNCell(internal_state_size)
state_series, current_state = tf.nn.static_rnn(cell, input_series, init_state)

# for current_input in input_series:
#     current_input = tf.reshape(current_input, [batch_size, number_of_sensors])
#
#     input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)
#
#     next_state = tf.tanh(tf.add(tf.matmul(input_and_state_concatenated, W_in), b_in))
#     state_series.append(next_state)
#     current_state = next_state

logits_series = [tf.add(tf.matmul(state, W_out), b_out) for state in state_series]
losses = [LossFunctions.diff(output_series, logits) for logits in logits_series]

total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    trainInput = TrainInput.TrainInput()

    for epoch_index in range(1):
        x, y = trainInput.get_chain_train_data(batch_size, number_of_sensors, number_of_efectors)
        _current_state = np.zeros((batch_size, internal_state_size))

        for batch_index in range(batch_size):
            start_index = batch_index * backpropagation_lenght
            end_index = start_index + backpropagation_lenght

            batchX = x[:, :, start_index:end_index]
            batchY = y[:, :, start_index:end_index]
            print(batchX)
            print(batchY)
            print(_current_state)
            _total_loss, _train_step, _current_state_tmp, _output_series = sess.run(
                [total_loss, train_step, current_state, output_series],
                feed_dict={
                    x_placeholder: batchX,
                    y_placeholder: batchY,
                    init_state: _current_state
                }
            )

            _current_state = np.array(_current_state_tmp)

            loss_list.append(_total_loss)

            if batch_index%100 == 0:
                print("step", batch_index, "loss", _total_loss)
