import numpy as np
import h5py as h5
import keyboard
from g27 import WheelStateDetector

class HumanAgent(object):
    def __init__(self, dim_states, use_logitech_g27 = False):
        self.filename = 'train_data/stateactionfile.h5'
        self.dim_states = dim_states
        self.use_logitech_g27 = use_logitech_g27
        self.prevAccel = 0
        self.prevSteer = 0
        self.prevBrake = 0
        self.prev_rpm = None
        self.prevGear = 0

        self.file = h5.File(self.filename, 'r+')
        dataset_name = "sa" + str(self.file.attrs['data_count'])
        self.dataset = self.file.create_dataset(dataset_name, (dim_states, 33)) #33 is dim of state action vector

        if use_logitech_g27:
            self.wheel_state_detector = WheelStateDetector()
            self.wheel_state_detector.start()

    def act(self, ob, reward, done, step):
        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # angle <-PI; PI>
        # trackPos <-1; 1>, (-inf; -1) u (1; +inf) - out of track

        focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, angle, trackPos = ob

        action = []

        if not self.use_logitech_g27:
            keys = self.getKeys()
            action.append(self.steer(keys[1], keys[2]))
            action.append(self.speed(keys[3]))
            action.append(self.gear(rpm))
            action.append(self.brake(keys[0]))
        else:
            action, clutch = self.wheel_state_detector.get_action()
            print(action)

        actualAS = []

        actualAS.extend(action)
        actualAS.extend([speedX, speedY, speedZ, rpm.tolist()])#, opponents, rpm, track, wheelSpinVel])
        actualAS.extend(wheelSpinVel.tolist())
        actualAS.extend(track.tolist())
        actualAS.extend([angle, trackPos])

        npArrayActualAS = np.array(actualAS)

        self.writeToFile(npArrayActualAS, step)

        return action # set action

    def writeToFile(self, AS, step):
        self.dataset[step] = AS


    def steer(self, left=0, right=0):
        steer = self.prevSteer

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

        self.prevSteer = steer

        return steer

    def speed(self, up=0):
        accel = self.prevAccel
        if up is 1:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        self.prevAccel = accel
        return accel

    def gear(self, rpm):
        gear = self.prevGear

        if self.prevGear == 0 and rpm > 2000:
            gear = 1
        elif self.prevGear == 1 and rpm > 7000:
            gear = 2
        elif self.prevGear == 2 and rpm > 7000:
            gear = 3
        elif self.prevGear == 3 and rpm > 7000:
            gear = 4
        elif self.prevGear == 4 and rpm > 7000:
            gear = 5
        elif self.prevGear == 5 and rpm > 7000:
            gear = 6
        elif self.prevGear > 1 and rpm < 3000:
            gear -= 1

        self.prevGear = gear
        return gear

    def brake(self, down = 0):
        brake = self.prevBrake
        if down is 1:
            brake += 0.1
            if brake > 1:
                brake = 1.0
        else:
            brake -= 0.2
            if brake < 0:
                brake = 0.0

        self.prevBrake = brake
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

    def reset(self):
        print("reset dataset")

    def next_dataset(self):
        self.file.attrs['data_count'] += 1
        dataset_name = "sa" + str(self.file.attrs['data_count'])
        self.dataset = self.file.create_dataset(dataset_name, (self.dim_states, 31))

    def end(self, acceptLastEpisode = True):
        self.wheel_state_detector.stop()
        if acceptLastEpisode:
            self.file.attrs['data_count'] += 1
            print("race" + str(self.file.attrs['data_count']) + "finished")
        else:
            self.file.__delitem__("sa" + str(self.file.attrs['data_count']))
        self.file.close()