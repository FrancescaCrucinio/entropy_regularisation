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
@rimport ks as rks;
# custom packages
using entropy_reg;

# set seed
Random.seed!(1234);

R"""
library(tictoc)
library(fDKDE)
library(readxl)
library(ks)

# get contaminated data
sucrase_Carter1981 <- read_excel("deconvolution/sucrase_Carter1981.xlsx");
W <- sucrase_Carter1981$Pellet;
n <- length(W);

# Normal error distribution
errortype="norm";
varU <- var(W)/4;
sigU <- sqrt(varU/2);

# DKDE
# Delaigle's estimators
# KDE for mu
h <- 1.06*sqrt(var(W))*n^(-1/5);
muKDE <- kde(W, h = h);
muKDEy <- muKDE$estimate;
muKDEx <- muKDE$eval.points;
dx <- muKDEx[2] - muKDEx[1];
bw <- muKDE$h;

# DKDE-pi
tic()
hPI <- PI_deconvUknownth4(W,errortype,varU,sigU);
fdec_hPI <- fdecUknown(muKDEx,W,hPI,errortype,sigU,dx);
toc()

# DKDE-cv
tic()
hCV <- CVdeconv(W, errortype,sigU);
fdec_hCV <- fdecUknown(muKDEx,W,hCV,errortype,sigU,dx);
toc()
"""

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
    hatMu= zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
# number of particles
Nparticles = 200;
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
alpha = 0.3;

tWGF = @elapsed begin
x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, sigU);
end
println("WGF done, $tWGF")

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
plot(EWGF)


reconstructions = [@rget(muKDEy) @rget(fdec_hPI) @rget(fdec_hCV) KDEyWGF[Niter, :]];
p = plot(@rget(muKDEx), reconstructions, lw = 2, label = ["KDE" "DKDE-pi" "DKDE-cv" "WGF"],
    legendfontsize = 15, tickfontsize = 10, color = [:gray :red :orange :blue])
# savefig(p, "sucrase.pdf")
