module DatasetMaker

import PyPlot
import Distributions


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
    mu = 0
    sigma = 0.15
    d = Distributions.Normal(mu, sigma)

    # add the fluctuations
    ys = [y + rand(d) for y in ys]

    return xs, ys
end


end
