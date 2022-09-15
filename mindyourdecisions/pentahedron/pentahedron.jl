using Optim
ENV["MPLBACKEND"] = "tkagg"
using PyPlot
using LinearAlgebra
using Statistics

"""
θ is the angle between PQ and PU, φ is the angle between PU and PO
so that θ = π/3 - φ.
"""

const ℓ = 10.0::Float64

ℓ1(φ) = ℓ√3/sin(2π/3-φ)
ℓ2(φ) = begin
    Δ = sin(φ)/sin(2π/3-φ)
    return ℓ√(1-2Δ+4Δ^2)
end

f(φ) = ℓ1(φ) + ℓ2(φ)

sol = optimize(f, 0.0, π/3; rel_tol=1e-10, abs_tol=1e-10)
φᵒ = sol.minimizer

φ = range(0.0, π/3; length=10001)

fig = figure(1)
fig.clf()
ax = subplot(1,1,1)
ax.plot(φ*180/π, f.(φ), linewidth=2)
ax.plot(φᵒ*180/π, f(φᵒ), marker="*", markersize=15, markeredgecolor="red", markerfacecolor="green")
ax.set_xlabel(L"φ \;\; [deg]", fontsize=15)
ax.set_ylabel(L"|PT|", fontsize=15)
