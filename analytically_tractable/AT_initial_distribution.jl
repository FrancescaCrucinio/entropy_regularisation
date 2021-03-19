push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using StatsPlots;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using entropy_reg;

# Compare initial distributions

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.045^2;
sigmaRho = 0.043^2;
sigmaMu = sigmaRho + sigmaK;
rho(x) = pdf.(Normal(0.5, sqrt(sigmaRho)), x);
mu(x) = pdf.(Normal(0.5, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);

# dt and number of iterations
dt = 1e-03;
Niter = 1000;
# samples from μ(y)
M = 500;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = 0.05;

# exact minimiser
variance, _  = AT_exact_minimiser(sigmaK, sigmaMu, alpha);
# initial distributions
x0 = [0.0*ones(1, Nparticles); 0.5*ones(1, Nparticles);
    1*ones(1, Nparticles); rand(1, Nparticles);
    0.5 .+ sqrt(variance)*randn(1, Nparticles);
    0.5 .+ sqrt(variance+0.01)*randn(1, Nparticles)];

E = zeros(Niter-1, size(x0, 1));
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : zero(x)
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = mu.(refY);
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(K.(KDEx, refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end
for i=1:size(x0, 1)
    ### WGF
    x, _ = wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0[i, :], M);
    # KDE
    KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
    E[:, i] = mapslices(psi, KDEyWGF, dims = 2);
end

p = plot(2:Niter, E, yaxis = :log, lw = 2, labels = ["delta 0" "delta 0.5" "delta 1" "U(0, 1)" "solution" "solution + noise"], legend = :outerright)
# savefig(p,"initial_distribution_E.pdf")
