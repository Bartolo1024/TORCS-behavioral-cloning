import numpy as np
import h5py as h5
import keyboard
from g27 import WheelStateDetector


class HumanAgent(object):
    def __init__(self, dim_states, use_logitech_g27=False):
        self.dim_states = dim_states
        self.use_logitech_g27 = use_logitech_g27
        self.prev_accel = 0
        self.prev_steer = 0
        self.prev_brake = 0
        self.prev_rpm = None
        self.prev_gear = 0

        self.data = np.zeros(shape=(dim_states, 33), dtype=np.float)

        if use_logitech_g27:
            self.wheel_state_detector = WheelStateDetector()
            self.wheel_state_detector.start()

    def act(self, ob, reward, done, step):
        # print('reward in actual state is %s' % reward)
        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # angle <-PI; PI>
        # trackPos <-1; 1>, (-inf; -1) u (1; +inf) - out of track

        focus, speed_x, speed_y, speed_z, opponents, rpm, track, wheelSpinVel, angle, trackPos = ob

        action = []

        if not self.use_logitech_g27:
            keys = self.getKeys()
            action.append(self.steer(keys[1], keys[2]))
            action.append(self.speed(keys[3]))
            action.append(self.gear(rpm))
            action.append(self.brake(keys[0]))
        else:
            action, clutch = self.wheel_state_detector.get_action()

        actual_as = []

        actual_as.extend(action)
        #actual_as.extend(clutch)
        actual_as.extend([speed_x, speed_y, speed_z, rpm.tolist()])#, opponents, rpm, track, wheelSpinVel])
        actual_as.extend(wheelSpinVel.tolist())
        actual_as.extend(track.tolist())
        actual_as.extend([angle, trackPos])
        self.data[step] = np.array(actual_as)

        return action# set action

    def reset(self):
        print("reset dataset")

    def next_dataset(self):
        self.file.attrs['data_count'] += 1
        dataset_name = "sa" + str(self.file.attrs['data_count'])
        self.dataset = self.file.create_dataset(dataset_name, (self.dim_states, 33))

    def end(self, accept_last_episode=True):
        self.wheel_state_detector.stop()
        if accept_last_episode:
            filename = 'train_data/stateactionfile.h5'
            file = h5.File(filename, 'r+')
            dataset_name = "sa" + str(file.attrs['data_count'])
            file.create_dataset(dataset_name, data=self.data) #33 is dim of state action vector
            file.attrs['data_count'] += 1
            print("race" + str(file.attrs['data_count']) + "finished")
            file.close()

    ### KEYS SECTION
    def steer(self, left=0, right=0):
        steer = self.prev_steer

        if right is 1 and left is 0:
            steer += 0.1
            if steer > 1:
                steer = 1.0
        elif right is 0 and left is 1:
            steer -= 0.1
            if steer < -1:
                steer = -1.0
        else:
            steer = 0

        self.prev_steer = steer

        return steer

    def speed(self, up=0):
        accel = self.prev_accel
        if up is 1:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        self.prev_accel = accel
        return accel

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

    def brake(self, down = 0):
        brake = self.prev_brake
        if down is 1:
            brake += 0.1
            if brake > 1:
                brake = 1.0
        else:
            brake -= 0.2
            if brake < 0:
                brake = 0.0

        self.prev_brake = brake
        return brake

    def getKeys(self):
        keys = []
        if keyboard.is_pressed('up'):
            keys.insert(0, 1)
        else:
            keys.insert(0, 0)
        if keyboard.is_pressed('left'):
            keys.insert(0, 1)
        else:
            keys.insert(0, 0)
        if keyboard.is_pressed('right'):
            keys.insert(0, 1)
        else:
            keys.insert(0, 0)
        if keyboard.is_pressed('space') or keyboard.is_pressed('down'):
            keys.insert(0, 1)
        else:
            keys.insert(0, 0)
        return keys