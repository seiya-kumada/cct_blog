#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import pymc3 as pymc
import pickle

if __name__ == '__main__':

    # load model
    with open('my_model.pkl', 'rb') as buff:
        data = pickle.load(buff)

    model, trace = data['model'], data['trace']
    ws = trace.get_values('ws', burn=5000, combine=False)
    print(type(ws))
    print(len(ws))
    print(type(ws[0]))
    print(ws[0].shape)
