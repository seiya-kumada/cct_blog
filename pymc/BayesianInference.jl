#!/usr/bin/env julia

include("Utils.jl")
include("DatasetMaker.jl")
include("Params.jl")

import DatasetMaker
import Utils
import Params


function draw_uncertainty(oxs, oys, sigmas, n, color)
    upper_bounds = oys + n * sigmas
    lower_bounds = oys - n * sigmas
    PyPlot.fill_between(oxs, lower_bounds, upper_bounds, alpha=0.3, label="[-$(n)σ,+$(n)σ]", facecolor=color)
end


function draw_curves(oxs, oys, oys_ground_truth, xs, ys, sigmas, i)
    PyPlot.title("Bayesian Inference")
 
    # draw predictive curve
    PyPlot.plot(oxs, oys, label="predictive curve", linestyle="dashed")
   
    # draw original curve
    PyPlot.plot(oxs, oys_ground_truth, label="original curve")
    
    # draw observed dataset
    PyPlot.scatter(xs, ys, label="observed dataset")

    # draw uncertainties
    draw_uncertainty(oxs, oys, sigmas, 3, "red")
    draw_uncertainty(oxs, oys, sigmas, 2, "yellow")
    draw_uncertainty(oxs, oys, sigmas, 1, "green")
    
    PyPlot.ylim(-0.1, 5.5)
    PyPlot.legend(loc="best")
    PyPlot.savefig(joinpath(Params.OUTPUT_DIR, "bayes_$(i).png"))
    # PyPlot.show()
    PyPlot.clf()
end


function main()
    # generate observed dataset
    xs, ys = DatasetMaker.make_observed_dataset(Params.RANGE, Params.N_SAMPLES)
    DatasetMaker.save_dataset(xs, ys, "./dataset.txt")

    for i in Params.MIN_DIM:Params.MAX_DIM
        # extend xs(vector) to matrix 
        xs_matrix = Utils.make_input_matrix(xs, i)

        # solve a problem by bayesian inference 
        s, w = Utils.make_solution(xs_matrix, ys, i)
        println("w", w)  
        println("s", s)

        # predict curve for oxs
        oxs = linspace(0, Params.RANGE, Params.N_STEPS)
        oxs_matrix = Utils.make_input_matrix(oxs, i)
        oys = oxs_matrix * w

        # 
        # println("length(oys)", length(oys))
        # for i in 1:100
        #     println(oys[i])
        # end

        # make original curve for oxs
        oys_ground_truth = DatasetMaker.original_curve.(oxs)

        # calculate sigma by bayesian inference
        sigmas = sqrt.(Utils.calculate_inv_lambda_in_bayesian(s, oxs, i))

        # draw curves
        draw_curves(oxs, oys, oys_ground_truth, xs, ys, sigmas, i)
    end
end

main()
