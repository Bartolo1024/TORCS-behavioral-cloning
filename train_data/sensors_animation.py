import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class SensorsAnimation(object):

    NUMBER_OF_RADIAL_SENSORS = 19
    track_list = ['track_%s' % ind for ind in range(0, 19)]

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_ylim(0, 1)
        self.theta = np.linspace(0.0, np.pi, num=self.NUMBER_OF_RADIAL_SENSORS)
        self.l, = self.ax.plot([], [])


    def update(self, i):
        """perform animation step"""
        df = self.dataframe[self.track_list]
        sensors = df.loc[i]
        self.l.set_data(self.theta, sensors)
        bars = self.ax.bar(self.theta, sensors, width=np.ones(self.NUMBER_OF_RADIAL_SENSORS)/self.NUMBER_OF_RADIAL_SENSORS, bottom=0.0)
        for r, bar in zip(sensors, bars):
            bar.set_facecolor(plt.cm.viridis(r / 10.))
            bar.set_alpha(0.5)
        return self.l,

    def run_animation(self):
        ani = FuncAnimation(self.fig, self.update, frames=10000, interval=100, blit=True)
        plt.show()