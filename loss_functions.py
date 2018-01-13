import tensorflow as tf


def diff(value, target_value):
    return tf.abs(tf.subtract(value, target_value))


def pow_loss_function(target_steering, steering_normalized, target_acceleration, acceleration_normalized,
                      lambda_parm=0.95, pow=12):
    steering_loss = tf.pow(tf.multiply(diff(steering_normalized, target_steering), (1 / (1 - lambda_parm))), pow)
    acceleration_loss = tf.multiply(tf.subtract(acceleration_normalized, target_acceleration), lambda_parm)

    loss_function = tf.reduce_mean(tf.add(steering_loss, acceleration_loss))

    return loss_function


def log_loss_function(target_steering, steering_normalized, target_acceleration, acceleration_normalized,
                      lambda_parm=0.9):
    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)

    # precission limit
    # steering_diff = tf.maximum(diff(target_steering, steering_normalized), 0.001)
    # acceleration_diff = tf.maximum(diff(target_acceleration, acceleration_normalized), 0.01)
    # brake_diff = tf.maximum(diff(target_brake, brake_normalized), 0.01)

    steering_loss = tf.log(steering_diff)
    acceleration_loss = tf.log(acceleration_diff)

    # lambda
    steering_loss_parametrized = tf.multiply(steering_loss, lambda_parm)
    acceleration_brake_parametrized = tf.multiply(acceleration_loss, (1 - lambda_parm))

    loss_function = tf.reduce_mean(tf.add(steering_loss_parametrized, acceleration_brake_parametrized))

    return loss_function


def exp_loss_function(target_steering, steering_normalized, target_acceleration, acceleration_normalized,
                      lambda_parm=0.9):
    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)

    steering_loss = tf.exp(tf.multiply(steering_diff, 1 / (1 - lambda_parm)))
    acceleration_loss = tf.exp(tf.multiply(acceleration_diff, 1 / lambda_parm))

    loss_function = tf.reduce_mean(tf.add(steering_loss, acceleration_loss))

    return loss_function


def exp_log_loss_function(target_steering, steering_normalized, target_acceleration, acceleration_normalized,
                          lambda_parm=0.9):
    steering_diff = diff(target_steering, steering_normalized)
    acceleration_diff = diff(target_acceleration, acceleration_normalized)

    steering_loss = tf.add(
        tf.exp(tf.multiply(steering_diff, (1 / (1 - lambda_parm)))),
        tf.log(tf.maximum(steering_diff, 0.0005))
    )
    acceleration_loss = tf.exp(tf.multiply(acceleration_diff, 1 / lambda_parm))

    loss_function = tf.reduce_mean(tf.add(steering_loss, acceleration_loss))
    return loss_function
