push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using entropy_reg;

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
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
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
    return kl-a*ent;
end

# dt and number of iterations
dt = 1e-03;
Niter = 200;

# samples from h(y)
M = 500;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = [0.01 0.1 0.5 1 1.1 1.5];
a = alpha[1];
E = zeros(Niter, length(alpha));
f_approx = zeros(length(KDEx), length(alpha));
for i=1:length(alpha)
    x0 = 0.5 .+ randn(1, Nparticles)/10;
    # run WGF
    x, _ =  wgf_AT_tamed(Nparticles, dt, Niter, alpha[i], x0, M);
    a = alpha[i];
    KDEyWGF = mapslices(phi, x, dims = 2);
    f_approx[:, i] = KDEyWGF[end, :];
    E[:, i] = mapslices(psi, KDEyWGF, dims = 2);
end

iterations = repeat(1:Niter, outer=[6, 1]);
solution = rho.(KDEx);


p1 = plot(KDEx, solution, lw = 2, legendfontsize = 12, tickfontsize = 10, color = :black, label = "true density")
plot!(p1, KDEx, f_approx, lw = 2, label = ["alpha = 0.01" "alpha = 0.1" "alpha = 0.5" "alpha = 1" "alpha = 1.1" "alpha = 1.5"])
# savefig(p1,"at_rho.pdf")

p2 = plot(1:Niter, E[:, 1], lw = 0, legendfontsize = 10, tickfontsize = 8, label = "")
plot!(p2, 1:Niter, E, lw = 2, label = "")
# savefig(p2,"at_E.pdf")
