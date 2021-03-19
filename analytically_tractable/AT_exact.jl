push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
@rimport ks as rks
# custom packages
using entropy_reg;

# Plot AT example and exact minimiser
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.045^2;
sigmaRho = 0.043^2;
sigmaMu = sigmaRho + sigmaK;
rho(x) = pdf.(Normal(0.5, sqrt(sigmaRho)), x);
mu(x) = pdf.(Normal(0.5, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end

# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = 0.1270;

x0 = 0.5 .+ randn(1, Nparticles)/10;
# x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0, M);
# KDE
# optimal bandwidth Gaussian
RKDE =  rks.kde(x[Niter, :], var"eval.points" = KDEx);
KDEyWGF = abs.(rcopy(RKDE[3]));
# exact minimiser
variance, _  = AT_exact_minimiser(sigmaK, sigmaMu, alpha);
ExactMinimiser = pdf.(Normal(0.5, sqrt(variance)), KDEx);

# solution
solution = rho.(KDEx);

p = plot(KDEx, solution, lw = 2, label = "true density", color = :black, legendfontsize = 10, tickfontsize = 8)
plot!(p, KDEx, ExactMinimiser, lw = 2, label = "exact minimiser", color = :red)
plot!(p, KDEx, KDEyWGF, lw = 2, label = "WGF reconstruction", color = :blue)
# savefig(p,"at_exact_min.pdf")
# savefig(p,"at_exact_eb.pdf")
