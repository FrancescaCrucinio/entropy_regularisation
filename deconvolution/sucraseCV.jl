push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using Distributions;
using Statistics;
using StatsBase;
using Random;
using StatsPlots;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using entropy_reg;

# set seed
Random.seed!(1234);

R"""
library(tictoc)
library(readxl)
library(ks)

# get contaminated data
sucrase_Carter1981 <- read_excel("deconvolution/sucrase_Carter1981.xlsx");
W <- sucrase_Carter1981$Pellet;
n <- length(W);

# Laplace error distribution
errortype="norm";
varU = var(W)/4;
sigU = sqrt(varU/2);
"""

a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = @rget muKDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = @rget muKDEy;
    refY = @rget muKDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = min(Nparticles, length(muSample));
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 20000;
# initial distribution
m0 = mean(muSample);
sigma0 = std(muSample);
x0 = m0 .+ sigma0*randn(1, Nparticles);
# regularisation parameter
alpha = range(0.1, stop = 0.5, length = 5);

# divide muSample into groups
L = 24;
muSample = reshape(muSample, (L, Int(length(muSample)/L)));

E = zeros(length(alpha), L);
Threads.@threads for i=1:length(alpha)
    @simd for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        # WGF
        x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, sigU);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
