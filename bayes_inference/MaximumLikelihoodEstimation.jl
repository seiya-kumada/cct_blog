#!/usr/bin/env julia

include("Utils.jl")
include("DatasetMaker.jl")
include("Params.jl")

import DatasetMaker
import Utils
import Params

xs, ys = DatasetMaker.make_observed_dataset(Params.RANGE, Params.N_SAMPLES)

input_matrix = Utils.make_input_matrix(xs)

# solution of maximum likelihood estimation
w = inv(input_matrix' * input_matrix) * input_matrix' * ys

# calculate inverse of lambda
sigma = sqrt(Utils.calculate_inv_lambda(w, xs, ys))
println("σ: $sigma")

# plot predictive curve
oxs = linspace(0, Params.RANGE, Params.N_STEPS)
oxs_matrix = Utils.make_input_matrix(oxs)
#@show size(oxs_matrix)

oys = oxs_matrix * w
#@show size(oys)

PyPlot.title("Maximum Likelihood Estimation")
PyPlot.scatter(xs, ys, label="observed dataset")
oys_ground_truth = DatasetMaker.original_curve.(oxs)


upper_bounds = [v + sigma for v in oys]
lower_bounds = [v - sigma for v in oys]

PyPlot.plot(oxs, oys, label="predictive curve")
PyPlot.fill_between(oxs, lower_bounds, upper_bounds, alpha=0.3, label="[-σ,+σ]", facecolor="green")



PyPlot.plot(oxs, oys_ground_truth, label="original curve")
PyPlot.legend(loc="best")
PyPlot.show()

