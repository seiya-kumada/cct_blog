#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Node:
    COUNTER = 0

    def __init__(self, m, n, values):
        self.m = m
        self.n = n
        self.values = values
        if n != 1:
            assert((m % n != 0) and (n % m != 0))
        # print("{} {},{}".format(Node.COUNTER, m, n))
        values.append((m, n))
        Node.COUNTER += 1

    def generate(self, upper_bound):
        if Node.COUNTER < upper_bound:
            m = self.m
            n = self.n
            node1 = Node(2 * m - n, m, self.values)
            node2 = Node(2 * m + n, m, self.values)
            node3 = Node(m + 2 * n, n, self.values)
            node1.generate(upper_bound)
            node2.generate(upper_bound)
            node3.generate(upper_bound)


class Generator:

    def generate(self, upper_bound):
        values = []
        node = Node(2, 1, values)
        node.generate(upper_bound)

        Node.COUNTER = 0
        node = Node(3, 1, values)
        node.generate(upper_bound)
        return values


if __name__ == "__main__":
    gen = Generator()
    values = gen.generate(100)
    print(len(values))
    for m, n in values:
        print(m, n)
