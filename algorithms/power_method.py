# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: power_method.py
@time: 2018/8/27 11:17

    求解lexrank用到的数值计算方法
"""

import numpy as np
from scipy.sparse.csgraph import connected_components


def _power_method(transition_matrix, increase_power=True):
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        return eigenvector

    transition = transition_matrix.transpose()

    while True:
        eigenvector_next = np.dot(transition, eigenvector)

        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next

        eigenvector = eigenvector_next

        if increase_power:
            transition = np.dot(transition, transition)


def connected_nodes(matrix):
    """
    从表示图的矩阵计算联通分量
    :param matrix:np.array
    :return:list:[list:int] 将所有的节点编号分组
    """

    _, labels = connected_components(matrix)

    groups = []

    for tag in np.unique(labels):
        group = np.where(labels == tag)[0]
        groups.append(group)

    return groups


def stationary_distribution(
    transition_matrix,
    increase_power=True,
    normalized=True,
):
    size = len(transition_matrix)
    distribution = np.zeros(size)

    grouped_indices = connected_nodes(transition_matrix)

    for group in grouped_indices:  # 对于每个连接分量分别进行计算，从而防止无法收敛的问题。 TODO:与原论文中添加扰动的方法对比，看看哪个效果好?
        t_matrix = transition_matrix[np.ix_(group, group)]
        eigenvector = _power_method(t_matrix, increase_power=increase_power)
        distribution[group] = eigenvector

    if normalized:
        distribution /= size

    return distribution


if __name__ == '__main__':
    t = np.ones([5,5])
    o = connected_nodes(t)