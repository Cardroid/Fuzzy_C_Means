import random
import math
from copy import deepcopy
from typing import List, Tuple

from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt


COLORS = []

PLOT_NUM = 0


def plot_helper():
    global PLOT_NUM

    PLOT_NUM += 1

    return PLOT_NUM


class data_point:
    def __init__(self, center_point_count: int, x: float = 0, y: float = 0) -> None:
        self.__membership = [0.0 for _ in range(center_point_count)]
        self.__x = x
        self.__y = y

    def __repr__(self):
        return f"({self.x}, {self.y}) [{self.group}] {self.membership}"

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value

    @property
    def membership(self):
        return self.__membership

    @property
    def group(self):
        group_idx = 0
        max_ms = 0
        for idx, ms in enumerate(self.__membership):
            if ms > max_ms:
                max_ms = ms
                group_idx = idx
        return group_idx


def generate_data_point(data_count: int, center_point_count: int) -> Tuple[List[data_point], Tuple[float, float, float, float]]:
    data_list = [data_point(center_point_count) for _ in range(data_count)]

    X, targets = make_blobs(
        n_samples=data_count,
        centers=center_point_count,
        n_features=2,
        cluster_std=0.6,
    )

    l, t, r, b = 0, 0, 0, 0
    for sample, data, target in zip(X, data_list, targets):
        data.x = sample[0]
        data.y = sample[1]

        l = min(l, data.x)
        r = max(r, data.x)

        b = min(b, data.y)
        t = max(t, data.y)

        data.membership[target] = 1.0

    return data_list, l, t, r, b


def generate_center_point(center_point_count: int, l: int, t: int, r: int, b: int) -> List[Tuple[int, int]]:
    center_point_list = [[random.randint(round(l), round(r)), random.randint(round(b), round(t))] for _ in range(center_point_count)]

    return center_point_list


def plot_show(data_list: List[data_point], center_point_list: List[List[float]], fig_num: int):
    plt.figure(fig_num)
    plt.cla()

    x_list = [p.x for p in data_list]
    y_list = [p.y for p in data_list]
    group_list = [COLORS[p.group] for p in data_list]

    plt.scatter(x_list, y_list, c=group_list)

    center_point_x_list = [p[0] for p in center_point_list]
    center_point_y_list = [p[1] for p in center_point_list]
    # group_list = [COLORS[i] for i in range(len(center_point_list))]

    plt.scatter(center_point_x_list, center_point_y_list, c="black", s=100, marker="^")

    plt.tight_layout()


def FCM(data_list: List[data_point], center_point_list: List[List[float]], m: float = 2.0):
    # FIT

    for data in data_list:
        total_dist = 0
        center_point_dist_list = [[0] for _ in range(len(center_point_list))]
        for idx, (c_x, c_y) in enumerate(center_point_list):
            temp_calc = math.sqrt((c_x - data.x) ** 2 + (c_y - data.y) ** 2) ** (2 / (m - 1))
            center_point_dist_list[idx].append(temp_calc)
            total_dist += temp_calc

        for m_idx in range(len(data.membership)):
            temp_calc = sum(center_point_dist_list[m_idx]) / total_dist
            data.membership[m_idx] = 0 if temp_calc == 0 else temp_calc**-1

    # MOVE

    for c_idx, center_point in enumerate(center_point_list):
        c_total_membership = 0
        c_x_total_membership = 0
        c_y_total_membership = 0

        for data in data_list:
            temp_calc = data.membership[c_idx] ** m
            c_x_total_membership += temp_calc * data.x
            c_y_total_membership += temp_calc * data.y
            c_total_membership += temp_calc

        center_point[0] = c_x_total_membership / c_total_membership
        center_point[1] = c_y_total_membership / c_total_membership


def main():
    global COLORS

    data_count = 200
    center_point_count = 5

    COLORS.extend(random.sample(["blue", "green", "red", "cyan", "magenta", "yellow"], center_point_count))

    data_list, l, t, r, b = generate_data_point(data_count, center_point_count)
    center_point_list = generate_center_point(center_point_count, l, t, r, b)

    fig_num = plot_helper()

    plot_show(data_list, center_point_list, fig_num=fig_num)

    plt.show(block=False)
    plt.pause(3)

    while True:
        FCM(data_list, center_point_list)
        plot_show(data_list, center_point_list, fig_num=fig_num)
        plt.pause(1)


if __name__ == "__main__":
    main()
