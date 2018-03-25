#!/usr/bin/env julia

import PyPlot
include("DatasetMaker.jl")
include("Params.jl")
import DatasetMaker


xs, ys = DatasetMaker.make_observed_dataset(Params.RANGE, Params.N_SAMPLES)

PyPlot.scatter(xs, ys, label="observed dataset")

# also draw the original curve
oxs = linspace(0, Params.RANGE, Params.N_STEPS)
oys = DatasetMaker.original_curve.(oxs)
PyPlot.plot(oxs, oys, label="original curve")
PyPlot.legend(loc="best")
PyPlot.show()

