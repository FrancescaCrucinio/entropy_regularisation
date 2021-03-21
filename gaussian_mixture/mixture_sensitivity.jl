push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using entropy_reg;
using samplers;

# set seed
Random.seed!(1234);

# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);

a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3])), rcopy(RKDE[4]);
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), KDEx)/3 +
            pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), KDEx)/3;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Normal.(refY, 0.045), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# parameters for WGF
# dt and number of iterations
dt = 1e-03;
Niter = 200;
# number of particles
Nparticles = 500;
# regularisation parameters
alpha = range(0, stop = 1, length = 100);
# initial distribution
x0 = 0.5 .+ randn(1, Nparticles)/10;
# samples from h(y)
M = 500;
muSample = Ysample_gaussian_mixture(100000);

Nrep = 100;
E = zeros(length(alpha), Nrep);
ise = zeros(length(alpha), Nrep);
variance = zeros(length(alpha), Nrep);
for i=1:length(alpha)
    a = alpha[i];
    for j=1:Nrep
    x = wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, a, x0, muSample, M);
    # KL
    KDE, h = phi(x[Niter, :]);
    E[i, j] = psi(KDE);
    ise[i, j] = var(rho.(KDEx) .- KDE);
    variance[i, j] = h + mean(x[Niter, :].^2) - mean(x[Niter, :])^2;
    println("$i, $j")
    end
end

Eavg = mean(E, dims = 2);
iseavg = mean(ise, dims = 2);
varavg = mean(variance, dims = 2);

p = plot(alpha, Eavg, lw = 2, label = "", legendfontsize = 12, tickfontsize = 10)
# savefig(p,"mixture_sensitivity_E.pdf")
