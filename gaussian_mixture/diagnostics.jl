#=
Diagnostics for approximations of f
OUTPUTS
1 - mean
2 - variance
3 - mean squared error
4 - mean integrated squared error
5 - entropy
INPUTS
'f' true f (function handle)
'x' sample points in the domain of f
'y' estimated value of f at sample points
=#
function diagnostics(f, x, y)
    #  mean
    m =  Statistics.mean(x, weights(y));
    # variance
    v = var(x, weights(y), corrected = false);
    # exact f
    trueF = f.(x);
    # compute MISE for f
    difference = (trueF .- y).^2;
    mise = mean(difference);
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(y .* log.(y)));
    return m, v, difference, mise, ent
end
