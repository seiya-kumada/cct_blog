#!/usr/bin/env julia

include("Utils.jl")
include("DatasetMaker.jl")
include("Params.jl")

import DatasetMaker
import Utils
import Params


function main()
    # generate observed dataset
    xs, ys = DatasetMaker.make_observed_dataset(Params.RANGE, Params.N_SAMPLES)

    # extend xs(vector) to matrix
    xs_matrix = Utils.make_input_matrix(xs, Params.M)

    # solve a problem by map estimation
    s, w = Utils.make_solution(xs_matrix, ys, Params.M)

    # predict curve for oxs
    oxs = linspace(0, Params.RANGE, Params.N_STEPS)
    oxs_matrix = Utils.make_input_matrix(oxs, Params.M)
    oys = oxs_matrix * w

    # make original curve
    oys_ground_truth = DatasetMaker.original_curve.(oxs)

    # calculate sigma 
    sigma = sqrt(Utils.calculate_inv_lambda(w, xs, ys, Params.M))
    println("σ: $sigma")

    # draw curves
    Utils.draw_curves("MAP Estimation", oxs, oys, oys_ground_truth, xs, ys, sigma, "./map.png")
end


main()
