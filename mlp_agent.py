import numpy as np
import tensorflow as tf
import model as mod
import normalization as norm
import os
import shared_preferences

class NeuralAgent(object):
    network_dirpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], "saved_network/")
    def __init__(self, max_steps):
        self.filename = 'stateactionfile.h5'
        self.max_steps = max_steps
        self.prev_rpm = None
        self.prev_gear = 0
        self.prevTrack = np.zeros(20)

        number_of_sensors = shared_preferences.number_of_sensors
        number_of_efectors = shared_preferences.number_of_efectors
        layers_shapes = shared_preferences.layers_shapes

        self.x = tf.placeholder(tf.float32, [None, number_of_sensors], name='state')

        # model
        normalized_input = norm.normalize_input(self.x)
        unnormalized_output = mod.mlp_model(normalized_input, number_of_sensors, number_of_efectors, layers_shapes)

        self.y = norm.normalize_output(unnormalized_output)

        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        self.restore_network()

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

        print(state)

        sensors_input = [np.array(state)]

        print(sensors_input)

        action = self.sess.run(self.y, feed_dict={self.x: sensors_input})

        steer = action[0]
        acc = action[1]
        acceleration = acc if acc > 0 else 0.2
        gearSignal = self.gear(rpm)
        brake = acc if acc < 0 else 0

        print([steer, acceleration, gearSignal, brake])

        return [steer, acceleration, gearSignal, brake] # set action

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

    def restore_network(self):
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00050one_race_p1_with_othermodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00005one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00005one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00005one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00005one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_100_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_200_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00050one_race_s1_model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00050one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00050one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00050one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00050one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_100_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_200_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00005one_racemodel.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00005model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00005model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00005model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00005model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00050model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00050model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00050model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00050model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00500model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_1000_lambda_param_0.900_learning_rate_0.00500model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_2500_lambda_param_0.900_learning_rate_0.00500model.ckpt")
        # self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_5000_lambda_param_0.900_learning_rate_0.00500model.ckpt")
        self.saver.restore(self.sess, self.network_dirpath + "mlp/batch_size_500_lambda_param_0.900_learning_rate_0.00050one_race_p1_with_othermodel.ckpt")
