import tensorflow as tf

def pow_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.05, pow = 12):
    loss_function = tf.reduce_sum(
        tf.add(tf.add(tf.pow(tf.multiply(tf.subtract(steering_normalized, target_steering), (1 / (lambda_parm))), pow),
                      tf.multiply(tf.abs(tf.subtract(acceleration_normalized, target_acceleration)), lambda_parm)),
               tf.multiply(tf.abs(tf.subtract(brake_normalized, target_brake)), lambda_parm)))
    return loss_function

def log_loss_function(target_steering, steering_normalized, target_brake, brake_normalized, target_acceleration, acceleration_normalized, lambda_parm = 0.9):
    loss_function = tf.reduce_mean(tf.add(
        tf.multiply(
            tf.log(
                tf.abs(
                    tf.subtract(target_steering, steering_normalized)
                )),
            lambda_parm),
        tf.multiply(
            tf.add(
                tf.log(
                    tf.abs(tf.subtract(target_brake, brake_normalized))
                ),
                tf.log(
                    tf.abs(tf.subtract(target_acceleration, acceleration_normalized))
                )),
            (1 - lambda_parm)
        )))
    return loss_function