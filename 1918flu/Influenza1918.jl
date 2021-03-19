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
@rimport ks as rks;
# custom packages
using entropy_reg;
include("RL.jl")

# set seed
Random.seed!(1234);

# fitted Gaussian approximating K
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
R"""
library(incidental)
library(tictoc)

# death counts
death_counts <- spanish_flu$Philadelphia

# RIDE estimator
tic()
Philadelphia_model <- fit_incidence(
  reported = spanish_flu$Philadelphia,
  delay_dist = spanish_flu_delay_dist$proportion)
toc()
RIDE_reconstruction <- Philadelphia_model$Chat/sum(Philadelphia_model$Chat)
RIDE_incidence <- Philadelphia_model$Ihat
"""
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# shuffle sample
shuffle!(muSample);
# x axis = time (122 days)
KDEx = 1:length(muCounts);
# KDE for μ
RKDE = rks.kde(muSample, var"eval.points" = KDEx);
muKDEy = abs.(rcopy(RKDE[3]));

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
    trueMu = muKDEy;
    refY = KDEx;
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
# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 3000;
# initial distribution
x0 = sample(muSample, Nparticles, replace = false) .- 9;
# regularisation parameter
alpha = 0.009;
runtimeWGF = @elapsed begin
# run WGF
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M);
end
# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
p = plot(EWGF);

# RL
# initial distribution
rho0 = [muCounts[9:end]; zeros(8, 1)];
# delay distribution
R"""
K_prop <- spanish_flu_delay_dist$proportion
K_day <- spanish_flu_delay_dist$days
"""
K_prop =@rget K_prop;
K_day = Int.(@rget K_day);

KDisc = eps()*ones(length(muCounts), length(muCounts));
for i=1:length(muCounts)
    for j=1:length(muCounts)
        if (i - j >= 1 && i - j<= length(K_day))
            KDisc[i, j] = K_prop[i - j];
        end
    end
end
runtimeRL = @elapsed begin
rhoCounts = RL(KDisc, muCounts, 200, rho0);
end
# recovolve WGF
refY = KDEx;
delta = refY[2] - refY[1];
KDEyRec = zeros(length(refY), 1);
for i=1:length(refY)
    KDEyRec[i] = delta*sum(K.(KDEx, refY[i]).*KDEyWGF[Niter, :]);
end
KDEyRec = KDEyRec/sum(KDEyRec);

# recovolve RL
RLyRec = zeros(1, length(refY));
for i=1:length(refY)
    t = refY[i] .- KDEx;
    nonnegative = (t .>= 1) .& (t .<= 31);
        RLyRec[i] = delta*sum(K_prop[t[nonnegative]].*rhoCounts[200, nonnegative]);
end
RLyRec = RLyRec/sum(RLyRec);

# plot
estimators = [rhoCounts[200, :]/sum(rhoCounts[200, :]) @rget(RIDE_incidence)/sum(@rget(RIDE_incidence)) KDEyWGF[Niter, :]];
p1=plot(KDEx, estimators, lw = 2, label = ["RL" "RIDE" "WGF"],
    color = [:gray :blue :red], line=[:solid :solid :solid],
    legendfontsize = 15, tickfontsize = 10)
# savefig(p1,"1918flu_incidence.pdf")

reconvolutions = [RLyRec[:] @rget(RIDE_reconstruction) KDEyRec].*sum(muCounts);
p2=scatter(KDEx, muCounts, marker=:x, markersize=3, label = "reported cases", color = :black)
plot!(p2, KDEx, reconvolutions, lw = 2, label = ["RL" "RIDE" "WGF"],
    color = [:gray :blue :red], line=[:solid :solid :solid],
    legendfontsize = 15, tickfontsize = 10)
# savefig(p2,"1918flu_reconvolution.pdf")
