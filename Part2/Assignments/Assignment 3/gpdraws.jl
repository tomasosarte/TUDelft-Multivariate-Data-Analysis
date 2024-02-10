workdir = @__DIR__
cd(workdir)

using LinearAlgebra
using PDMats
using RCall
using Distributions
using DataFrames
using Plots
using GaussianProcesses

include("gpfuncs.jl")

### drawing from a Gaussian process
Nsim = 5
L = 100
X = range(-3.0,3.0;length=L)

ℓ = 1.5
C1 = gpprior(X,Nsim,K1(ℓ=ℓ);titlename="Draws square exponential kernel (ls $ℓ)")
covstructure(X,C1;titlename="Cov struct square exponential kernel (ls $ℓ)")

m = 1
C2 = gpprior(X,Nsim,K2(m=m);titlename="Draws Polynomial($m) kernel")
covstructure(X,C2;titlename="Cov struct polynomial($m) kernel")

Σ = .5
C3 = gpprior(X,Nsim,K3(Σ=Σ);titlename="Draws neuronal($Σ) kernel")
covstructure(X,C3;titlename="Cov struct neuronal($Σ) kernel")

### simple example of gp regression
f⁰(x) = 2.0 + 0.2*sin(30*x) + (x>0.7)
Xnew = 0.0:.005:1.0
# plot true signal
plot(Xnew,f⁰,label="true")

# generate data and superimpose scatterplot of data
σ = 0.2
n = 200
X = sort(rand(n))
y = f⁰.(X) + σ*randn(n)
scatter!(X,y,label="data",markersize=2.0)

# specify kernel
K = K1()

# compute posterior mean on grid specified by Xnew
C =[K(x,y) + K0(;σ=σ)(x,y)  for x in X, y in X]
C = PDMat((C + C')*0.5)
R = [K(x,y) + K0(;σ=σ)(x,y) for x in X, y in Xnew]
Cnew = [K(x,y) + K0(;σ=σ)(x,y) for x in Xnew, y in Xnew]
Σnew = Cnew - Xt_invA_X(C,R)
fnew_mean = R' * (C \ y)

plot!(Xnew, fnew_mean,label="fit")
