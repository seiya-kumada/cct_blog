include("Params.jl")

module DatasetMaker

import PyPlot
import Distributions
import Params

# the same results are always obtained
srand(1)

# original curve
function original_curve(x)
    x + sin(3x)
end

function make_observed_dataset(range, n_samples)
    xs = range * rand(n_samples)
    ys = original_curve.(xs)

    # define gaussian
    d = Distributions.Normal(Params.MU, Params.SIGMA)

    # add the fluctuations
    ys = [y + rand(d) for y in ys]

    return xs, ys
end


end
