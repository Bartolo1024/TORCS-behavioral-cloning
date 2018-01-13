import tensorflow as tf

def mlp_model(normalized_input, number_of_sensors, number_of_efectors, layers_shapes):
    import variables

    parmeters = variables.initialize_parmeters(number_of_sensors, number_of_efectors, layers_shapes)

    output = normalized_input

    for index in range(1, int(len(parmeters)/2) + 1):
        with tf.name_scope("layer" + str(index)):
            next = tf.add(tf.matmul(output, parmeters["W" + str(index)]), parmeters["b" + str(index)])

            output = tf.nn.tanh(next, name="tanh" + str(index))

    return output


def lstm_model(batch_size, input_series, number_of_sensors, internal_state_size, number_of_efectors, number_of_layers):
    from variables import initialize_lstm_parameters

    layers = tf.placeholder(tf.float32, [number_of_layers, 2, batch_size, internal_state_size])
    layer_states_list = tf.unstack(layers, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(layer_states_list[index][0], layer_states_list[index][1])
         for index in range(number_of_layers)]
    )

    parameters = initialize_lstm_parameters(number_of_sensors, number_of_efectors, internal_state_size)

    # forward passes
    cells = []
    for _ in range(number_of_layers):
        cells.append(tf.nn.rnn_cell.LSTMCell(internal_state_size, state_is_tuple=True))

    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    state_series, current_state = tf.nn.static_rnn(multi_cell, input_series, initial_state=rnn_tuple_state)

    outputs_series = [tf.add(tf.matmul(state, parameters["W_out"]), parameters["b_out"]) for state in state_series]
    return outputs_series, current_state, layers





