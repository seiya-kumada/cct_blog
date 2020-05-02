#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Node:

    def __init__(self, m, n, values):
        self.m = m
        self.n = n
        self.values = values
        if n != 1:
            assert((m % n != 0) and (n % m != 0))
        # print("{},{}".format(m, n))
        values.append((m, n))

    def generate(self):
        m = self.m
        n = self.n
        node1 = Node(2 * m - n, m, self.values)
        node2 = Node(2 * m + n, m, self.values)
        node3 = Node(m + 2 * n, n, self.values)
        return node1, node2, node3


class Generator:

    def __init__(self, initial_m, initial_n):
        self.m = initial_m
        self.n = initial_n

    def generate(self, upper_bound):
        values = []

        node0 = Node(self.m, self.n, values)
        node1, node2, node3 = node0.generate()

        nodes = []
        nodes.append(node1)
        nodes.append(node2)
        nodes.append(node3)

        c = 0
        while True:
            new_nodes = []
            for node in nodes:
                node1, node2, node3 = node.generate()
                new_nodes.append(node1)
                new_nodes.append(node2)
                new_nodes.append(node3)
            nodes = new_nodes
            c += 1
            if c > upper_bound:
                break
        return values


if __name__ == "__main__":
    gen = Generator(2, 1)
    values = gen.generate(10)
    for m, n in values:
        print(m, n)

    gen = Generator(3, 1)
    values = gen.generate(10)
    for m, n in values:
        print(m, n)
