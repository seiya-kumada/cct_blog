#!/usr/bin/env julia

include("Params.jl")

module Utils
import Params
import PyPlot


function make_input_vector(x, m)
    [x^i for i in 0:m - 1]
end


function make_input_matrix(xs, m)
    mat = Matrix{Float64}(size(xs)[1], m)
    for (i, x) in enumerate(xs)
        mat[i, :] = make_input_vector(x, m)
    end
    mat
end


function calculate_inv_lambda(w, xs, ys, m)
    input_matrix = make_input_matrix(xs, m)
    pys = input_matrix * w
    dy = ys - pys
    dot(dy, dy) / size(ys)[1]
end


function calculate_inv_lambda_in_bayesian(s, xs, m)
    xs_matrix = make_input_matrix(xs, m)
    1 / Params.LAMBDA + diag(xs_matrix * s * xs_matrix')
end


function make_solution(matrix, ys, m)
    s= inv(Params.ALPHA * eye(m, m) + Params.LAMBDA * matrix' * matrix) 
    w = Params.LAMBDA * s * matrix' * ys
    s, w
end


function draw_uncertainty(oxs, oys, sigma, n, color)
    upper_bounds = [v + n * sigma for v in oys]
    lower_bounds = [v - n * sigma for v in oys]
    PyPlot.fill_between(oxs, lower_bounds, upper_bounds, alpha=0.3, label="[-$(n)σ,+$(n)σ]", facecolor=color)
end


function draw_curves(title, oxs, oys, oys_ground_truth, xs, ys, sigma, path)
    PyPlot.title(title * "(σ=$sigma)")

    # draw observed dataset
    PyPlot.scatter(xs, ys, label="observed dataset")

    # draw predictive curve
    PyPlot.plot(oxs, oys, label="predictive curve", linestyle="dashed")
    
    # draw uncertainty
    #draw_uncertainty(oxs, oys, sigma, 3, "red")
    #draw_uncertainty(oxs, oys, sigma, 2, "yellow")
    #draw_uncertainty(oxs, oys, sigma, 1, "green")
   
    PyPlot.ylim(-0.1, 5.5)
    PyPlot.plot(oxs, oys_ground_truth, label="original curve")
    PyPlot.legend(loc="best")
    PyPlot.savefig(path)
    PyPlot.clf()
    # PyPlot.show()
end


function calculate_model_evidence(xs, ys, w, s, m)
    a = Params.LAMBDA * dot(ys, ys)
    b = -Params.N_SAMPLES * log(Params.LAMBDA / 2pi) - m * log(Params.ALPHA)
    c = -w' * inv(s) * w -logdet(s)
    -0.5 * (a + b + c)
end


end
