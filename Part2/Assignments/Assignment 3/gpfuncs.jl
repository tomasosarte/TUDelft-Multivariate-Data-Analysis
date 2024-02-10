# Define some kernels

"""
    Independence kernel with standard deviation σ
"""
K0(;σ=1.0) = (x,y) -> σ*(abs(x-y)<10^(-6))


"""
    Square exponential kernel with variance v and length scale ℓ
"""
K1(;v=1.0,ℓ=.1) = (x,y) -> v*exp(-0.5*dot(x-y,x-y)/ℓ^2) + 10^(-8)*(norm(x-y)<10^(-6))

"""
    Polynomial kernel of degree m and variance v
"""
K2(; v=1.0, m=2) = (x,y) -> v*(1+dot(x,y))^m + 10^(-8)*(norm(x-y)<10^(-6))

function K3(x,y;Σ)
    a(x,y,Σ) = 2.0 * dot(x,Σ*y)
    2.0 * asin(a(x,y,Σ)/sqrt((1+a(x,x,Σ))*(1+a(y,y,Σ))))/π + 10^(-8)*(norm(x-y)<10^(-6))
end
"""
    Neuronal kernel with covariance matrix Σ
"""
K3(;Σ=10) = (x,y) -> K3(x,y;Σ=Σ)


"""
    X: grid on which we simulate Gaussian process
    C: matrix with C[i,j] = cov(f(x_i), f(x_j))

    Make tile plot of C
"""
function covstructure(X, C; titlename="Covariance structure of Gaussian process prior")
    L= length(X)
    dd = DataFrame(z=vec(C.mat), x=repeat(X,inner=L), y=repeat(X,outer=L))
    @rput dd
    @rput titlename
    R"""
    p <- ggplot(dd, aes(x=x, y=y, fill= z)) + geom_tile()    + ggtitle(titlename)
    pdf(paste0(titlename,".pdf"),width=4.5, height=4)
        show(p)
    dev.off()
    """
end

"""
    X: grid on which we simulate Gaussian process
    Nsim: number of draws from prior
    K: kernel specification
"""
function gpprior(X,Nsim, K;titlename="draws from Gaussian process prior")
    # construct Gram matrix
    K_ = [K(x,y)  for x in X, y in X] + ScalMat(length(X), 10^(-6))
    K__= PDMat((K_ + K_')*0.5) # avoid numerical round-off induced nonsymmetry

    # draw from MvNormal

    f = rand(MvNormal(zeros(size(K__)[1]), K__), Nsim)

    # plotting
    d = DataFrame(hcat(X,f))
    st = vcat("x",  ["$i" for i in 1:Nsim])
    rename!(d, Symbol.(st))
    @rput d
    @rput titlename
    R"""
    library(ggplot2)
    library(tidyverse)
    p <- d %>% gather(key="sample",value="value",-x) %>% ggplot(aes(x=x,y=value,colour=sample))+
            geom_path() + theme_light() + ggtitle(titlename)
    pdf(paste0(titlename,".pdf"),width=6,height=3)
        show(p)
    dev.off()
    """
    K__
end
