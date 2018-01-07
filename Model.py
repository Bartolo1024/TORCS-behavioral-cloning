import tensorflow as tf
import Variables

def mlp_model(normalized_input, number_of_sensors, number_of_efectors, layers_shapes):

    parmeters = Variables.initialize_parmeters(number_of_sensors, number_of_efectors, layers_shapes)

    output = normalized_input

    for index in range(1, int(len(parmeters)/2) + 1):
        with tf.name_scope("layer" + str(index)):
            next = tf.add(tf.matmul(output, parmeters["W" + str(index)]), parmeters["b" + str(index)])

            output = tf.nn.tanh(next, name="tanh" + str(index))

    return output

def rnn_model():
    print("todo")





