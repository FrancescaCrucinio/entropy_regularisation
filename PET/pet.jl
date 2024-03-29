push!(LOAD_PATH, "C:/Users/Francesca/Desktop/entropy_regularisation/modules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;
using Distances;
using KernelDensity;
# custom packages
using entropy_reg;
using samplers;

# set seed
Random.seed!(1234);

# Shepp Logan phantom
phantom = readdlm("PET/phantom.txt", ',', Float64);
phantom = reverse(phantom, dims=1);
pixels = size(phantom);
# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64);
# sinogram = reverse(sinogram, dims=1);
# number of angles
nphi = size(sinogram, 2);
# angles
phi_angle = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));
xi = xi/maximum(xi);

# grid
X1bins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
X2bins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX1 = repeat(X1bins, inner=[pixels[2], 1]);
gridX2 = repeat(X2bins, outer=[pixels[1] 1]);
KDEeval = [gridX1 gridX2];

# function computing KDE
function phi(t)
    B = kde((t[1:Nparticles], t[(Nparticles+1):(2Nparticles)]));
    Bpdf = pdf(B, X1bins, X2bins);
    return abs.(Bpdf[:]);
end
# function computing entropy
function psi_ent(t)
    t = t./maximum(t);
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
end
# function computing KL
function psi_kl(t)
    # kl
    trueMu = sinogram;
    refY1 = phi_angle;
    refY2 = xi;
    # approximated value
    delta1 = refY1[2] - refY1[1];
    delta2 = refY2[2] - refY2[1];
    hatMu = zeros(length(refY2), length(refY1));
    # convolution with approximated ρ
    # this gives the approximated value
    for i=1:length(refY2)
        for j=1:length(refY1)
            hatMu[i, j] = sum(pdf.(Normal.(0, sigma), KDEeval[:, 1] * cos(refY1[j]) .+
                KDEeval[:, 2] * sin(refY1[j]) .- refY2[i]).*t);
        end
    end
    hatMu = hatMu/maximum(hatMu);
    kl = kl_divergence(trueMu[:], hatMu[:]);
    return kl;
end

# WGF
# dt and number of iterations
dt = 1e-02;
Niter = 100;
# samples from h(y)
M = 20000;
# number of particles
Nparticles = 20000;
# regularisation parameter
# matching smcems entropy
alpha = 0.01;
# cross validation
# alpha = 1e-04;
# variance of normal describing alignment
sigma = 0.02;
# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);
runtime = @elapsed begin
    x1, x2 = wgf_pet_tamed(Nparticles, dt, Niter, alpha, muSample, M, sigma);
end
# KDE
KDEyWGF = mapslices(phi, [x2 x1], dims = 2);
# entropy
ent = mapslices(psi_ent, KDEyWGF, dims = 2);
# KL
KLWGF = mapslices(psi_kl, KDEyWGF, dims = 2);

# check convergence
plot(KLWGF)
phantom_ent = psi_ent(phantom);
plot(ent)
hline!([phantom_ent])
plot(KLWGF .- alpha * ent)

KDEyWGFfinal = KDEyWGF[end, :];

# ise
petWGF = reshape(KDEyWGFfinal, (pixels[1], pixels[2]));
petWGF = petWGF/maximum(petWGF);
var(petWGF .- phantom)
# relative error
rel_error = abs.(petWGF .- phantom);
positive = phantom .> 0;
rel_error[positive] = rel_error[positive]./phantom[positive];

# plot
p1 = heatmap(phantom, color = :inferno, aspect_ratio = 1, axis = false, colorbar = false, size=(600, 600));
# savefig(p1, "phantom.pdf")
p2 = heatmap(petWGF, color = :inferno, aspect_ratio = 1, axis = false, colorbar = false, size=(600, 600));
# savefig(p2, "pet_cv.pdf")
p3 = heatmap(rel_error, color = :inferno, aspect_ratio = 1, axis = false, colorbar = false, size=(600, 600));
# savefig(p3, "pet_cv_re.pdf")
