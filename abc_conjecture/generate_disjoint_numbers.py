#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Node:
    COUNTER = 0
    UPPER_BOUND = 100

    def __init__(self, m, n):
        self.node1_ = None
        self.node2_ = None
        self.node3_ = None
        self.m_ = m
        self.n_ = n
        if n != 1:
            assert((m % n != 0) and (n % m != 0))
        print("{},{}".format(m, n))
        Node.COUNTER += 1

    def generate(self):
        if Node.COUNTER < Node.UPPER_BOUND:
            m = self.m_
            n = self.n_
            self.node1_ = Node(2 * m - n, m)
            self.node2_ = Node(2 * m + n, m)
            self.node3_ = Node(m + 2 * n, n)
            self.node1_.generate()
            self.node2_.generate()
            self.node3_.generate()


if __name__ == "__main__":
    node = Node(2, 1)
    node.generate()

    Node.COUNTER = 0
    node = Node(3, 1)
    node.generate()
