module entropy_reg;

using Distributions;
using Statistics;
using LinearAlgebra;

using samplers;

export wgf_AT_tamed
export wgf_gaussian_mixture_tamed
export wgf_pet_tamed
export AT_exact_minimiser
export wgf_sucrase_tamed
export wgf_DKDE_tamed
export wgf_flu_tamed

#= WGF for analytically tractable example
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_AT_tamed(N, dt, Niter, alpha, x0, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # initialise a matrix drift storing the drift
    drift = zeros(Niter-1, N);

    for n=1:(Niter-1)
        # get samples from h(y)
        y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[n, i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift[n, :]./(1 .+ Niter^(-a) * abs.(drift[n, :])) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x, drift
end

#= WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_gaussian_mixture_tamed(N, dt, Niter, alpha, x0, muSample, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+  dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

#= WGF for positron emission tomography
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'muSample' sample from noisy image μ(y)
'M' number of samples from h(y) to be drawn at each iteration
'sigma' standard deviation for Normal describing alignment
'a' parameter for tamed Euler scheme
=#
function wgf_pet_tamed(N, dt, Niter, alpha, muSample, M, sigma, a)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # intial distribution
    x0 = rand(MvNormal([0, 0], 0.1*Diagonal(ones(2))), N);
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];
    for n=1:(Niter-1)
        # get sample from μ(y)
        muIndex = sample(1:size(muSample, 1), M, replace = true);
        y = muSample[muIndex, :];
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(y[j, 1]) .+
                    x2[n, :] * sin(y[j, 1]) .- y[j, 2])
                    );
        end
        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2]) .*
                    (x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2])/sigma^2;
            gradientX1 = prec .* cos.(y[:, 1]);
            gradientX2 = prec .* sin.(y[:, 1]);
            # keep only finite elements
            g1h = gradientX1./hN;
            g2h = gradientX2./hN;
            g1h[(!).(isfinite.(g1h))] .= 0;
            g2h[(!).(isfinite.(g2h))] .= 0;
            driftX1[i] = mean(g1h);
            driftX2[i] = mean(g2h);
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x1, x2
end

#= Exact minimiser for analytically tractable example
OUTPUTS
1 - variance
2 - KL divergence
INPUTS
'sigmaK' variance of kernel K
'sigmaMu' variance of data function μ
'alpha' regularisation parameter
=#
function AT_exact_minimiser(sigmaK, sigmaMu, alpha)
    variance  = (sigmaMu - sigmaK .+ 2*alpha*sigmaK .+
                sqrt.(sigmaK^2 + sigmaMu^2 .- 2*sigmaK*sigmaMu*(1 .- 2*alpha)))./
                (2*(1 .- alpha));
    KL = 0.5*log.((sigmaK .+ variance)/sigmaMu) .+ 0.5*sigmaMu./(sigmaK .+ variance) .- 0.5 .-
        0.5*alpha .* (1 .+ log.(2*pi*variance));
    return variance, KL
end

#= WGF for deconvolution with real data (sucrase example)
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
'sigU' parameter for error distribution
=#
function wgf_sucrase_tamed(N, dt, Niter, alpha, x0, muSample, M, a, sigU)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # get samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            # hN[j] = mean(pdf.(Normal.(x[n, :], sigU), y[j]));
            hN[j] = mean(pdf.(Laplace.(x[n, :], sigU), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            # gradient = pdf.(Normal.(x[n, i], sigU), y) .* (y .- x[n, i])/(sigU^2);
            gradient = pdf.(Laplace.(x[n, i], sigU), y) .* (-sign.(x[n, i] .- y)/sigU);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

#= WGF for deconvolution with simulated data and Laplace error
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
'sigU' parameter for error distribution
=#
function wgf_DKDE_tamed(N, dt, Niter, alpha, x0, muSample, M, a, sigU)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # get samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Laplace.(x[n, :], sigU), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Laplace.(x[n, i], sigU), y) .* (-sign.(x[n, i] .- y)/sigU);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end
#= WGF for Spanish flu data
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_flu_tamed(N, dt, Niter, alpha, x0, muSample, M, a)
   # initialise a matrix x storing the particles
   x = zeros(Niter, N);
   # initial distribution is given as input:
   x[1, :] = x0;

   for n=1:(Niter-1)
       # get samples from h(y)
       y = sample(muSample, M, replace = true);
       # Compute h^N_{n}
       hN = zeros(M, 1);
       for j=1:M
           hN[j] = mean(0.595*pdf.(Normal(8.63, 2.56), y[j] .- x[n, :]) +
                   0.405*pdf.(Normal(15.24, 5.39), y[j] .- x[n, :]))
       end

       # gradient and drift
       drift = zeros(N, 1);
       for i=1:N
           gradient = 0.595*pdf.(Normal(8.63, 2.56), y .- x[n, i]) .* (y .- x[n, i] .- 8.63)/(2.56^2) +
                   0.405*pdf.(Normal(15.24, 5.39), y .- x[n, i]) .* (y .- x[n, i] .- 15.24)/(5.39^2);
           drift[i] = mean(gradient./hN);
       end
       # update locations
       x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
   end
   return x
end

end
