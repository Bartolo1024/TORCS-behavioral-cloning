import numpy as np
import tensorflow as tf
import normalization as norm
import os
from model import lstm_model
import shared_preferences

class NeuralAgent(object):
    network_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "saved_network/lstm/")
    def __init__(self, max_steps):
        self.filename = 'stateactionfile.h5'
        self.max_steps = max_steps
        self.prev_rpm = None
        self.prev_gear = 0
        self.prevTrack = np.zeros(20)

        self.number_of_sensors = shared_preferences.number_of_sensors
        self.number_of_efectors = shared_preferences.number_of_efectors
        self.backpropagation_size = shared_preferences.backpropagation_size
        self.internal_state_size = shared_preferences.internal_state_size

        self.input = tf.placeholder(tf.float32, [1, self.number_of_sensors, self.backpropagation_size], name='state')
        self.sensors_input = np.zeros((self.backpropagation_size, self.number_of_sensors))

        # model
        normalized_input = norm.normalize_input(self.input)
        input_series = tf.unstack(normalized_input, axis=2)
        self.output_series, self.current_state, self.init_state = lstm_model(1, input_series,
                                                                             self.number_of_sensors,
                                                                             self.internal_state_size,
                                                                             self.number_of_efectors,
                                                                             shared_preferences.number_of_lstm_layers)

        unnormalized_output = self.output_series[-1]

        steering_normalized, acceleration_normalized, self.y = norm.normalize_output(unnormalized_output)

        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        self.saver.restore(self.sess, self.network_dirpath + "model.ckpt")

    def act(self, ob, reward, done, step):
        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.

        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, angle, trackPos = ob

        #keys = self.getKeys()

        if np.any(track < 0) == 1:
            track = self.prevTrack;

        self.prevTrack = track

        state = []

        state.extend([speedX, speedY, speedZ, rpm.tolist()])
        state.extend(wheelSpinVel.tolist())
        state.extend(track.tolist())
        state.extend([angle, trackPos])

        print(self.sensors_input)
        np.roll(self.sensors_input, self.number_of_sensors, axis=1)
        self.sensors_input[-1] = np.array(state)

        print(self.sensors_input)

        _current_state = np.zeros((shared_preferences.number_of_lstm_layers, 2, 1, shared_preferences.internal_state_size))

        _current_state, _output_series, action = self.sess.run(
            [self.current_state, self.output_series, self.y],
            feed_dict={
                self.input: [self.sensors_input.transpose()],
                self.init_state: _current_state
            }
        )

        steer = action[0][0]
        acc = action[0][1]
        acceleration = acc if acc > 0 else 0
        gearSignal = self.gear(rpm)
        brake = acc if acc < 0 else 0

        print([steer, acceleration, gearSignal, brake])

        acceleration, brake = 0.4, 0
        return [steer, acceleration, gearSignal, brake if acceleration < 0.1 else 0] # set action

    def gear(self, rpm):
        gear = self.prev_gear

        if self.prev_gear == 0 and rpm > 2000:
            gear = 1
        elif self.prev_gear == 1 and rpm > 7000:
            gear = 2
        elif self.prev_gear == 2 and rpm > 7000:
            gear = 3
        elif self.prev_gear == 3 and rpm > 7000:
            gear = 4
        elif self.prev_gear == 4 and rpm > 7000:
            gear = 5
        elif self.prev_gear == 5 and rpm > 7000:
            gear = 6
        elif self.prev_gear > 1 and rpm < 3000:
            gear -= 1

        self.prev_gear = gear
        return gear

    def end(self, acceptLastEpisode = True):
        self.sess.close()
        print("end")