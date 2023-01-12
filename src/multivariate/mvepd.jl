"""
    MvEPD(p, μ, Σ)

The *Multivariate exponential power distribution* is a multidimensional generalization of the 
*exponential power distribution* (also known as generalized gaussian). The probability density function of
a d-dimensional multivariate exponential power distribution with mean vector ``\\boldsymbol{\\mu}`` and
covariance matrix ``\\boldsymbol{\\Sigma}`` is [1]:

```math
f(\\mathbf{x}; p, \\boldsymbol{\\mu}, \\boldsymbol{\\Sigma}) = k |\\boldsymbol{\\Sigma}|^{1/2}}
\\exp \\left( - \\frac{1}{2} \\big[(\\mathbf{x} - \\boldsymbol{\\mu})^T \\Sigma^{-1} (\\mathbf{x} - \\boldsymbol{\\mu})\\big]^p \\right),
```
where
```math
k = \\frac{d \\Gamma(d/2)}{\\pi^{\\frac{1}{2}} \\Gamma(1+d/(2p))2^{1+\\frac{d}{2p}}}
```

[1] Zhy, D. and V. Zinde-Walsh (2009). Properties and estimation of asymmetric exponential power distribution. _Journal of econometrics_, 148(1):86-96, 2009.
"""
abstract type AbstractMvEpd <: ContinuousMultivariateDistribution end

struct GenericMvEpd{T<:Real, Cov<:AbstractPDMat, Mean<:AbstractVector} <: AbstractMvEpd
    p::T
    dim::Int
    μ::Mean
    Σ::Cov

    function GenericMvEpd{T, Cov, Mean}(p::T, dim::Int, μ::Mean, Σ::AbstractPDMat{T}) where {T, Cov, Mean}
        p > zero(p) || error("p must be positive")
        new{T, Cov, Mean}(p, dim, μ, Σ)
    end
end

function GenericMvEpd(p::T, μ::Mean, Σ::Cov) where {Cov<:AbstractPDMat, Mean<:AbstractVector,T<:Real}
    d = length(μ)
    size(Σ) == (d,d) || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent"))
    R = Base.promote_eltype(T, μ, Σ)
    S = convert(AbstractArray{R}, Σ)
    m = convert(AbstractArray{R}, μ)
    GenericMvEpd{R, typeof(S), typeof(m)}(R(p), d, m, S)
end

function GenericMvEpd(p::Real, Σ::AbstractPDMat)
    R = Base.promote_eltype(p, Σ)
    GenericMvEpd(p, zeros(R, size(Σ)), Σ)
end

function convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd) where T <:Real
    S = convert(AbstractArray{T}, d.Σ)
    m = convert(AbstractArray{T}, d.μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(d.p), d.dim, m, S)
end
Base.convert(::Type{GenericMvEpd{T}}, d::GenericMvEpd{T}) where {T<:Real} = d

function convert(::Type{GenericMvEpd{T}}, p, dim, μ::AbstractVector, Σ::AbstractPDMat) where T<:Real
    S = convert(AbstractArray{T}, Σ)
    m = convert(AbstractArray{T}, μ)
    GenericMvEpd{T, typeof(S), typeof(m)}(T(p), dim, m, S)
end

MvEpd(p::Real, μ::Vector{<:Real}, Σ::PDMat) = GenericMvEpd(p, μ, Σ)
MvEpd(p::Real, Σ::PDMat) = GenericMvEpd(p, Σ)
MvEpd(p::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = GenericMvEpd(p, μ, PDMat(Σ))
MvEpd(p::Real, Σ::Matrix{<:Real}) = GenericMvEpd(p, PDMat(Σ))

length(d::GenericMvEpd) = d.dim
params(d::GenericMvEpd) = (d.p, d.dim, d.μ, d.Σ)
sqmahal(d::GenericMvEpd, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)
cov(d::GenericMvEpd) = (2^(1/d.p)*gamma((d.dim+2)/(2*d.p))/(d.dim*gamma(d.dim/(2*d.p))))*d.Σ
var(d::GenericMvEpd) = (2^(1/d.p)*gamma((d.dim+2)/(2*d.p))/(d.dim*gamma(d.dim/(2*d.p))))*diag(d.Σ)
invscale(d::GenericMvEpd) = Matrix(inv(d.Σ))

insupport(d::AbstractMvEpd, x::AbstractVector{T}) where {T<:Real} =
    length(d) == length(x) && all(isfinite, x)

Base.eltype(::Type{<:GenericMvEpd{T}}) where {T} = T

function mvepd_const(d::AbstractMvEpd)
    H = convert(eltype(d), pi^(-d.dim/2))
    log(H) + log(d.dim) + loggamma(d.dim/2) - loggamma(1+d.dim/(2*d.p)) - (1+d.dim/(2*d.p))*log(2)
end

function logpdf(d::AbstractMvEpd, x::AbstractVector{T}) where T<:Real
    k = mvepd_const(d)
    mvepd_const(d) -0.5 * logdet(d.Σ) -0.5*sqmahal(d, x)^d.p
end

pdf(d::AbstractMvEpd, x::AbstractVector{<:Real}) = exp(logpdf(d, x))

sampler(d::GenericMvEpd) = Mvepdsampler(d.p, d.μ, d.Σ)

_rand!(rng::AbstractRNG, d::GenericMvEpd, x::AbstractMatrix) = _rand!(rng, sampler(d), x)
_rand!(rng::AbstractRNG, d::GenericMvEpd, x::AbstractVector) = _rand!(rng, sampler(d), x)
