#!/usr/bin/env julia

import PyPlot
include("DatasetMaker.jl")
import DatasetMaker

N_SAMPLES = 20
RANGE = 4

xs, ys = DatasetMaker.make_observed_dataset(RANGE, N_SAMPLES)

PyPlot.scatter(xs, ys, label="observed dataset")

# also draw the original curve
oxs = linspace(0, RANGE, 100)
oys = DatasetMaker.original_curve.(oxs)
PyPlot.plot(oxs, oys, label="original curve")
PyPlot.show()

