import matplotlib.pyplot as plt
from data_parser import Data
import matplotlib.gridspec as gridspec
import numpy as np


class Graphs:
    @staticmethod
    def create_fd_scatter_graph():
        aircraft, ground = Graphs.average_data()
        gridspec.GridSpec(2,2)
        x_axis = range(len(aircraft))

        plt.subplot2grid((2,2),(0,0))
        plt.plot(x_axis, aircraft, 'r,')
        plt.title("Aircraft")

        plt.subplot2grid((2,2),(0,1))
        plt.plot(x_axis, ground, 'b,')
        plt.title("Ground")

        plt.subplot2grid((2,2),(1,0), colspan=2)
        plt.plot(x_axis, ground, 'b,', x_axis, aircraft, 'r.')
        plt.title("Combined")

        plt.show()

    @staticmethod
    def average_data():
        aircraft, ground = Data.create_fd_arrays()

        aircraft_average = []
        total = 0
        length = len(aircraft[0])
        for i in range(length):
            for j in aircraft:
                total += j[i]
            aircraft_average.append(total / length)
            total = 0

        ground_average = []
        total = 0
        length = len(ground[0])
        for i in range(length):
            for j in ground:
                total += j[i]
            ground_average.append(total / length)
            total = 0

        return aircraft_average, ground_average

Graphs.create_fd_scatter_graph()
