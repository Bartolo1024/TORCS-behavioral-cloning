import tensorflow as tf

def model(normalized_input, W1, W2, W3, W4, W5, W6, b1, b2, b3, b4, b5, b6):
    with tf.name_scope("layer_input"):
        input_layer = tf.add(tf.matmul(normalized_input, W1), b1)
    with tf.name_scope("layer_h1"):
        hidden_layer_a = tf.nn.relu(tf.add(tf.matmul(input_layer, W2), b2))
    with tf.name_scope("layer_h2"):
        hidden_layer_b = tf.nn.relu(tf.add(tf.matmul(hidden_layer_a, W3), b3))
    with tf.name_scope("layer_h3"):
        hidden_layer_c = tf.nn.relu(tf.add(tf.matmul(hidden_layer_b, W4), b4))
    with tf.name_scope("layer_h4"):
        hidden_layer_d = tf.nn.relu(tf.add(tf.matmul(hidden_layer_c, W5), b5))
    with tf.name_scope("output"):
        unnormalized_output = tf.add(tf.matmul(hidden_layer_d, W6), b6)
    return unnormalized_output
