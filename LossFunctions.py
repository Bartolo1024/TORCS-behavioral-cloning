import tensorflow as tf

def diff(value, target_value):
    return tf.abs(tf.subtract(value, target_value))

def add3(val1, val2, val3):
    return tf.add(tf.add(val1, val2), val3)

def pow_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.95, pow = 12):
    steering_loss = tf.pow(tf.multiply(diff(steering_normalized, target_steering), (1 / (1 - lambda_parm))), pow)
    acceleration_loss = tf.multiply(tf.subtract(acceleration_normalized, target_acceleration), lambda_parm)
    brake_loss = tf.multiply(tf.subtract(brake_normalized, target_brake), lambda_parm)

    loss_function = tf.reduce_mean(add3(steering_loss, acceleration_loss, brake_loss))

    return loss_function

def log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.9):

    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)
    brake_diff = diff(target_brake, brake_normalized)

    # precission limit
    #steering_diff = tf.maximum(diff(target_steering, steering_normalized), 0.001)
    #acceleration_diff = tf.maximum(diff(target_acceleration, acceleration_normalized), 0.01)
    #brake_diff = tf.maximum(diff(target_brake, brake_normalized), 0.01)

    steering_loss = tf.log(steering_diff)
    acceleration_loss = tf.log(acceleration_diff)
    brake_loss = tf.log(brake_diff)

    #lambda
    steering_loss_parametrized = tf.multiply(steering_loss, lambda_parm)
    acceleration_brake_parametrized = tf.multiply(tf.add(acceleration_loss, brake_loss), (1 - lambda_parm))

    loss_function = tf.reduce_mean(tf.add(steering_loss_parametrized, acceleration_brake_parametrized))

    return loss_function

def exp_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.99, batch = 1000):
    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)
    brake_diff = diff(target_brake, brake_normalized)

    steering_loss = tf.exp(tf.multiply(steering_diff, 1 / (1 - lambda_parm)))
    acceleration_loss = tf.exp(tf.multiply(acceleration_diff, 1 / lambda_parm))
    brake_loss = tf.exp(tf.multiply(brake_diff, 1 / lambda_parm))

    loss_function = tf.reduce_mean(add3(steering_loss, acceleration_loss, brake_loss))

    return loss_function

def exp_log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.99, batch = 1000):
    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)
    brake_diff = diff(target_brake, brake_normalized)

    steering_loss = tf.add(
                    tf.exp(tf.multiply(steering_diff, (1 / (1 - lambda_parm)))),
                    tf.maximum(tf.log(tf.multiply(steering_diff, 1)), 0.0005)
                )
    acceleration_loss = tf.exp(tf.multiply(acceleration_diff, 1 / lambda_parm))
    brake_loss = tf.exp(tf.multiply(brake_diff, 1 / lambda_parm))

    loss_function = tf.reduce_mean(add3(steering_loss, acceleration_loss, brake_loss))
    return loss_function