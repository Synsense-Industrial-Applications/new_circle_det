import csv
import os
import threading
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty, Full, Queue as ThreadQueue

import numpy as np


class point_withchannel:
    def __init__(self, c, x, y):
        self.c = c
        self.x = x
        self.y = y


class ChannelHelper:
    """通用 channel/坐标映射工具，和具体 Speck 网络配置解耦。"""

    def __init__(self, input_channel, input_size, output_size):
        self.input_channel = input_channel
        self.input_size = input_size
        self.output_size = output_size

    def get_tuple(self, input_tuple, mode_flage=0):
        c, x, y = input_tuple
        if mode_flage == 0:
            if self.input_size == self.output_size:
                return (c, x, y)
            if self.input_size > self.output_size:
                ratio = self.input_size // self.output_size
                local_pos = (x % ratio) * ratio + (y % ratio)
                out_c = c * (ratio**2) + local_pos
                out_x = x // ratio
                out_y = y // ratio
                return (out_c, out_x, out_y)

            ratio = self.output_size // self.input_size
            out_c = c // (ratio**2)
            local_pos = c % (ratio**2)
            j = local_pos // ratio
            k = local_pos % ratio
            out_x = x * ratio + j
            out_y = y * ratio + k
            return (out_c, out_x, out_y)

        if mode_flage == 1:
            if self.input_size == self.output_size:
                return (c, x, y)
            if self.input_size > self.output_size:
                ratio = self.input_size // self.output_size
                in_c = c // (ratio**2)
                local_pos = c % (ratio**2)
                j = local_pos // ratio
                k = local_pos % ratio
                in_x = x * ratio + j
                in_y = y * ratio + k
                return (in_c, in_x, in_y)

            ratio = self.output_size // self.input_size
            local_pos = (x % ratio) * ratio + (y % ratio)
            in_c = c * (ratio**2) + local_pos
            in_x = x // ratio
            in_y = y // ratio
            return (in_c, in_x, in_y)

        raise ValueError(f"unknown mode_flage: {mode_flage}")

    def generate_conv_mapping(self):
        if self.input_size == self.output_size:
            raise ValueError("Input and output sizes are equal; no mapping needed.")

        if self.input_size > self.output_size:
            ratio = self.input_size // self.output_size
            output_channel = self.input_channel * (ratio**2)
            padding = 0
            stride = ratio
            kernel_size = ratio
            weights = np.zeros((output_channel, self.input_channel, kernel_size, kernel_size), dtype=np.int8)

            for in_c in range(self.input_channel):
                for j in range(ratio):
                    for k in range(ratio):
                        out_c = in_c * (ratio**2) + j * ratio + k
                        weights[out_c, in_c, j, k] = 1
            return padding, stride, kernel_size, weights

        ratio = self.output_size // self.input_size
        output_channel = self.input_channel // (ratio**2)
        padding = 0
        stride = 1
        kernel_size = ratio
        weights = np.zeros((output_channel, self.input_channel, kernel_size, kernel_size), dtype=np.int8)

        for out_c in range(output_channel):
            for j in range(ratio):
                for k in range(ratio):
                    in_c = out_c * (ratio**2) + j * ratio + k
                    weights[out_c, in_c, j, k] = 1
        return padding, stride, kernel_size, weights

    def caculate_cxy(self, x, y, radius):
        input_size = self.input_size
        output_size = self.output_size
        out_channel = input_size // output_size
        ret_x = (x - radius) // out_channel
        ret_y = (y - radius) // out_channel
        c = 0

        if x - radius < 0:
            remainder_x = np.abs(x - radius) % out_channel
            c += 0 if remainder_x == 0 else (out_channel - remainder_x) * out_channel
        else:
            c += ((x - radius) % out_channel) * out_channel

        if y - radius < 0:
            remainder_y = np.abs(y - radius) % out_channel
            c += 0 if remainder_y == 0 else (out_channel - remainder_y)
        else:
            c += (y - radius) % out_channel

        return c, ret_x, ret_y

    def get_kernel(self, kernel_matrix):
        input_size = self.input_size
        output_size = self.output_size
        out_channel = input_size // output_size
        kernel_size = len(kernel_matrix)
        print("kernel_size:", kernel_size)
        radius = kernel_size // 2
        caculate_size = out_channel + radius * 2

        adjacency_matrix = [
            [point_withchannel(*self.caculate_cxy(x, y, radius)) for y in range(caculate_size)]
            for x in range(caculate_size)
        ]

        shift_kernel_size = int(np.ceil(radius / out_channel) * 2 + 1)
        kernel_weight = np.zeros((out_channel**2, out_channel**2, shift_kernel_size, shift_kernel_size))
        center = int(np.ceil(radius / out_channel))
        for i in range(out_channel**2):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    temp_x = radius + i // out_channel
                    temp_y = radius + i % out_channel
                    source = adjacency_matrix[temp_x - (radius - j)][temp_y - (radius - k)]
                    target = adjacency_matrix[temp_x][temp_y]
                    kernel_weight[i, source.c, center + source.x - target.x, center + source.y - target.y] = kernel_matrix[j][k]
        return kernel_weight

    def convert_matrix_to_128(self, matrix_64, out_channel):
        return matrix_64[out_channel, :, :, :]

    def get_conv_shape(self):
        return self.input_channel, self.output_size, self.output_size

def save_events(evs, filename, columns=['layer', 'x', 'y', 'feature', 'timestamp']):
    if len(evs) == 0:
        print("no events")
        return
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([columns] + [[int(getattr(ev, _)) for _ in columns] for ev in evs])

def print_events(duration, layers,filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if buf:
        buf.get_events()
        time.sleep(duration)
        evs = buf.get_events()
        evs_layer  =  [[] for _ in range(14)]
        for ev in evs:
            evs_layer[ev.layer] += [ev]

        selected_events = [ev for ev in evs if ev.layer in layers]
        save_events(selected_events, filename)
    else:
        print("no buf")