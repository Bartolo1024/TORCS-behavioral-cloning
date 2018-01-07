import tensorflow as tf

def mlp_model(normalized_input, number_of_sensors, number_of_efectors, layers_shapes):
    import Variables

    parmeters = Variables.initialize_parmeters(number_of_sensors, number_of_efectors, layers_shapes)

    output = normalized_input

    for index in range(1, int(len(parmeters)/2) + 1):
        with tf.name_scope("layer" + str(index)):
            next = tf.add(tf.matmul(output, parmeters["W" + str(index)]), parmeters["b" + str(index)])

            output = tf.nn.tanh(next, name="tanh" + str(index))

    return output


def lstm_model(batch_size, input_series, number_of_sensors, internal_state_size, number_of_efectors):
    from Variables import initialize_lstm_parameters
    cell_state = tf.placeholder(tf.float32, [batch_size, internal_state_size])
    hidden_state = tf.placeholder(tf.float32, [batch_size, internal_state_size])
    init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

    parameters = initialize_lstm_parameters(number_of_sensors, number_of_efectors, internal_state_size)

    # forward passes
    cell = tf.nn.rnn_cell.BasicLSTMCell(internal_state_size, state_is_tuple=True)
    state_series, current_state = tf.nn.static_rnn(cell, input_series, init_state)

    outputs_series = [tf.add(tf.matmul(state, parameters["W_out"]), parameters["b_out"]) for state in state_series]
    return outputs_series, cell_state, hidden_state





