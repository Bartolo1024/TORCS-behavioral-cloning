import tensorflow as tf

def initialize_parmeters(number_of_sensors, number_of_efectors, layer_shapes):

    parameters = {}

    parameters["W1"] = tf.get_variable("W1", [number_of_sensors, layer_shapes[0]], initializer = tf.contrib.layers.xavier_initializer())
    parameters["b1"] = tf.get_variable("b1", [1, layer_shapes[0]], initializer = tf.zeros_initializer())
    #tf.summary.histogram("W1", parameters["W1"])
    #tf.summary.histogram("b1", parameters["b1"])

    for index, value in enumerate(layer_shapes[1:]):
        layer_shape_index = index + 1
        layer_parm_index = str(index + 2)
        parameters["W" + layer_parm_index] = tf.get_variable("W" + layer_parm_index, [layer_shapes[layer_shape_index - 1], layer_shapes[layer_shape_index]], initializer = tf.contrib.layers.xavier_initializer())
        parameters["b" + layer_parm_index] = tf.get_variable("b" + layer_parm_index, [1, layer_shapes[layer_shape_index]], initializer = tf.zeros_initializer())
        #tf.summary.histogram("W" + layer_parm_index, parameters["W" + layer_parm_index])
        #tf.summary.histogram("b" + layer_parm_index, parameters["b" + layer_parm_index])

    return parameters

def initialize_lstm_parameters(number_of_sensors, number_of_efectors, internal_state_size):
    parameters = {}

    parameters["W_out"] = tf.get_variable("W_out", [internal_state_size, number_of_efectors],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    parameters["b_out"] = tf.get_variable("b_out", [1, number_of_efectors],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)

    return parameters