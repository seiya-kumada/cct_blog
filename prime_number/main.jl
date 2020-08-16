#!/usr/bin/env julia
import PyPlot
const plt = PyPlot

THETAS = [
    14.1347,
    21.0220,
    25.0109,
    30.4249,
    32.9351,
    37.5862,
    40.9187,
    43.3271,
    48.0052,
    49.7738,
    52.9703,
    56.4462,
    59.3470,
    60.8318,
    65.1125,
]

MIN_X = 1.5
MAX_X = 15.0
N = 1000


if abspath(PROGRAM_FILE) == @__FILE__

    xs = range(MIN_X, MAX_X, length=N)
    for i in 1:length(THETAS)
        ys = zeros(Float64, N) 
        println("-- $(i) --")
        for t in THETAS[1:i]
            ys -= cos.(t * log.(xs)) ./ sqrt.(xs) 
        end
        
        max_y = maximum(ys)
        min_y = minimum(ys)
        plt.matplotlib.rcParams["font.size"] = 12
        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, label="N=$(i)")
        plt.xlabel("\$x\$")
        plt.ylabel("\$y\$")
        plt.xticks([2, 3, 5, 7, 8, 9, 11, 13], ["2", "3", "5", "7", "(8)", "(9)", "11", "13"])
        plt.vlines([2, 3, 5, 7, 11, 13], ymin=min_y, ymax=max_y, linestyles="dotted") 
        plt.legend(loc="upper right")
        path = "./images/julia/primes_$(lpad(string(i), 2, '0')).jpg"
        plt.savefig(path)
        plt.clf()
    end
    #println()
end
