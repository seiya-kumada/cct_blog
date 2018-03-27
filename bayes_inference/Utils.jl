#!/usr/bin/env julia

include("Params.jl")

module Utils
import Params
import PyPlot


function make_input_vector(x)
    [x^i for i in 0:Params.M - 1]
end


function make_input_matrix(xs)
    mat = Matrix{Float64}(size(xs)[1], Params.M)
    for (i, x) in enumerate(xs)
        mat[i, :] = make_input_vector(x)
    end
    mat
end


function calculate_inv_lambda(w, xs, ys)
    input_matrix = make_input_matrix(xs)
    pys = input_matrix * w
    dy = ys - pys
    dot(dy, dy) / size(ys)[1]
end


function calculate_inv_lambda_in_bayesian(s, xs)
    xs_matrix = make_input_matrix(xs)
    1 / Params.LAMBDA + diag(xs_matrix * s * xs_matrix')
end


function make_solution(matrix, ys)
    s= inv(Params.ALPHA * eye(Params.M, Params.M) + Params.LAMBDA * matrix' * matrix) 
    w = Params.LAMBDA * s * matrix' * ys
    s, w
end


function draw_uncertainty(oxs, oys, sigma)
    upper_bounds = [v + sigma for v in oys]
    lower_bounds = [v - sigma for v in oys]
    PyPlot.fill_between(oxs, lower_bounds, upper_bounds, alpha=0.3, label="[-σ,+σ]", facecolor="green")
end


function draw_curves(title, oxs, oys, oys_ground_truth, xs, ys, sigma, path)
    PyPlot.title(title)

    # draw observed dataset
    PyPlot.scatter(xs, ys, label="observed dataset")

    # draw predictive curve
    PyPlot.plot(oxs, oys, label="predictive curve")
    
    # draw uncertainty
    draw_uncertainty(oxs, oys, sigma)
    
    PyPlot.plot(oxs, oys_ground_truth, label="original curve")
    PyPlot.legend(loc="best")
    PyPlot.savefig(path)
    PyPlot.show()
end


function calculate_model_evidence(xs, ys, w, s)
    a = Params.LAMBDA * dot(ys, ys)
    b = -Params.N_SAMPLES * log(Params.LAMBDA / 2pi) - Params.M * log(Params.ALPHA)
    c = -w' * inv(s) * w -logdet(s)
    -0.5 * (a + b + c)
end


end
