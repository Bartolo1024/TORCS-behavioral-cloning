import tensorflow as tf

number_of_features_hidden_a = 27  #
number_of_features_hidden_b = 20  #
number_of_features_hidden_c = 15  #
number_of_features_hidden_d = 10  #
number_of_features_hidden_e = 10  #

def GetRandomWeights(number_of_sensors, number_of_efectors):
    W1 = tf.Variable(tf.truncated_normal([number_of_sensors, number_of_features_hidden_a], stddev=0.1), name="W1")
    W2 = tf.Variable(tf.truncated_normal([number_of_features_hidden_a, number_of_features_hidden_b], stddev=0.1),
                     name="W2")
    W3 = tf.Variable(tf.truncated_normal([number_of_features_hidden_b, number_of_features_hidden_c], stddev=0.1),
                     name="W3")
    W4 = tf.Variable(tf.truncated_normal([number_of_features_hidden_c, number_of_features_hidden_d], stddev=0.1),
                     name="W4")
    W5 = tf.Variable(tf.truncated_normal([number_of_features_hidden_d, number_of_features_hidden_e], stddev=0.1),
                     name="W5")
    W6 = tf.Variable(tf.truncated_normal([number_of_features_hidden_e, number_of_efectors], stddev=0.1), name="W6")
    return W1, W2, W3, W4, W5, W6


def GetNullBiases(number_of_sensors, number_of_efectors):
    b1 = tf.Variable(tf.zeros(number_of_features_hidden_a), name="b1")
    b2 = tf.Variable(tf.zeros(number_of_features_hidden_b), name="b2")
    b3 = tf.Variable(tf.zeros(number_of_features_hidden_c), name="b3")
    b4 = tf.Variable(tf.zeros(number_of_features_hidden_d), name="b4")
    b5 = tf.Variable(tf.zeros(number_of_features_hidden_e), name="b5")
    b6 = tf.Variable(tf.zeros(number_of_efectors), name="b6")
    return b1, b2, b3, b4, b5, b6