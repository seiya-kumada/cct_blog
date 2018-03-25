#!/usr/bin/env julia

include("Params.jl")

module Utils
import Params


function make_input_vector(x)
    [x^i for i in 0:Params.M - 1]
end


function make_input_matrix(xs)
        mat = Matrix{Float64}(size(xs)[1], Params.M)
    for (i, x) in enumerate(xs)
        mat[i, :] = make_input_vector(x)
    end
    return mat
end


function calculate_inv_lambda(w, xs, ys)
    input_matrix = make_input_matrix(xs)
    pys = input_matrix * w
    dy = ys - pys
    return dot(dy, dy) / size(ys)[1]
end


function calculate_inv_lambda_in_bayesian(s, xs)
    xs_matrix = make_input_matrix(xs)
    return 1 / Params.LAMBDA + diag(xs_matrix * s * xs_matrix')
end


#@show size(make_input_vector(2))
#@show make_input_vector(2)
#@show size(make_input_matrix([1,2,3]))
#@show make_input_matrix([1,2,3])

end
