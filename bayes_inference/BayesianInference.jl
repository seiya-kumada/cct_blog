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
s= inv(Params.ALPHA * eye(Params.M, Params.M) + Params.LAMBDA * input_matrix' * input_matrix) 
w = Params.LAMBDA * s * input_matrix' * ys

# calculate inverse of lambda
#sigma = sqrt(Utils.calculate_inv_lambda(w, xs, ys))
#println("σ: $sigma")

# plot predictive curve
oxs = linspace(0, Params.RANGE, Params.N_STEPS)
oxs_matrix = Utils.make_input_matrix(oxs)

oys = oxs_matrix * w

PyPlot.title("Bayesian Inference")
PyPlot.scatter(xs, ys, label="observed dataset")
oys_ground_truth = DatasetMaker.original_curve.(oxs)

sigmas = sqrt.(Utils.calculate_inv_lambda_in_bayesian(s, oxs))

upper_bounds_3 = oys + 3sigmas
lower_bounds_3 = oys - 3sigmas
PyPlot.fill_between(oxs, lower_bounds_3, upper_bounds_3, alpha=0.3, label="[-3σ,+3σ]", facecolor="red")


upper_bounds_2 = oys + 2sigmas
lower_bounds_2 = oys - 2sigmas
PyPlot.fill_between(oxs, lower_bounds_2, upper_bounds_2, alpha=0.3, label="[-2σ,+2σ]", facecolor="yellow")


upper_bounds_1 = oys + sigmas
lower_bounds_1 = oys - sigmas
PyPlot.fill_between(oxs, lower_bounds_1, upper_bounds_1, alpha=0.3, label="[-σ,+σ]", facecolor="green")

PyPlot.plot(oxs, oys, label="predictive curve")


PyPlot.plot(oxs, oys_ground_truth, label="original curve")
PyPlot.legend(loc="best")
PyPlot.show()

