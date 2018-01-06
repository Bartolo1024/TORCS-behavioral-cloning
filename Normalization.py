import tensorflow as tf

def normalize_input(input):
    speedX, speedY, speedZ, rpm, wheelSpinVel, track, angle, trackPos = tf.split(input, [1, 1, 1, 1, 4, 19, 1, 1], 1)
    normalized_speed_x = tf.divide(speedX, 500)
    normalized_speed_y = tf.divide(speedY, 500)
    normalized_speed_z = tf.divide(speedZ, 500)
    normalized_rpm = tf.divide(rpm, 100)
    normalized_wheel_spin_velocity = tf.divide(wheelSpinVel, 1000)
    normalized_track = track
    normalized_angle = tf.divide(angle, 3.14)
    normalized_trackPos = trackPos
    normalized_input = tf.concat(
        [normalized_speed_x, normalized_speed_y, normalized_speed_z, normalized_rpm, normalized_wheel_spin_velocity,
         track, normalized_angle, normalized_trackPos],
        1)
    return normalized_input

def normalize_output(unnormalized_output):
    print(tf.shape(unnormalized_output))
    steering, acceleration, brake = tf.split(unnormalized_output, [1, 1, 1], 1)
    steering_normalized = tf.tanh(steering)  # steering values e(-1, 1)
    acceleration_normalized = tf.sigmoid(acceleration)  # acceleration values e(0, 1)
    brake_normalized = tf.sigmoid(brake)  # acceleration values e(0, 1)

    with tf.name_scope("normalized_output"):
        y = tf.concat([steering_normalized, acceleration_normalized, brake_normalized], 1)

    return steering_normalized, acceleration_normalized, brake_normalized, y