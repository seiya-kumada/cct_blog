#!/usr/bin/env julia

include("Utils.jl")
include("DatasetMaker.jl")
include("Params.jl")

import DatasetMaker
import Utils
import Params

MAX_DIM = 11

function main()
    # generate observed dataset
    xs, ys = DatasetMaker.make_observed_dataset(Params.RANGE, Params.N_SAMPLES)

    vs = []
    for i in 1:MAX_DIM
        # M-dependence: extend xs(vector) to matrix 
        xs_matrix = Utils.make_input_matrix(xs, i)

        # solve a problem by bayesian inference 
        s, w = Utils.make_solution(xs_matrix, ys, i)

        v = Utils.calculate_model_evidence(xs, ys, w, s, i)
        push!(vs, v)
        println("$i: $v")
    end
    
    PyPlot.title("Model Evidence")
    xs = [i for i in 1:MAX_DIM] 
    PyPlot.plot(xs, vs)
    PyPlot.xlim(3, 11)
    PyPlot.ylim(-300, 0)
    PyPlot.savefig("model_evidence.png")
    PyPlot.xlabel("dimension")
    PyPlot.ylabel("model evidence")
    PyPlot.show()
end


main()
