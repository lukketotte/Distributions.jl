struct Mvepdsampler{T <: Real} <: Sampleable{Multivariate,Continuous}
  p::T
  μ::Vector{T}
  Σ::PDMat{T}
end

Mvepdsampler(p::Real, μ::Vector{<:Real}, Σ::PDMat{<:Real}) = Mvepdsampler(p, μ, Σ)

length(spl::Mvepdsampler) = length(spl.μ)

function runifsphere(rng::AbstractRNG, d::Int)
  mvnorm = rand(rng, Normal(), d)
  mvnorm ./ sqrt.(sum(mvnorm.^2))
end

function _rand!(rng::AbstractRNG, spl::Mvepdsampler, x::AbstractMatrix)
  dim,n = size(x)
  Σ = sqrt(spl.Σ)
  for i in 1:n
    R = rand(rng, Gamma(dim/(2*spl.p), 2))^(1/(2*spl.p))
    @inbounds x[:,i] = spl.μ + R*Σ*runifsphere(rng, dim)
  end
  x
end

function _rand!(rng::AbstractRNG, spl::Mvepdsampler, x::AbstractVector)
  dim = length(spl)
  R = rand(rng, Gamma(dim/(2*spl.p), 2))^(1/(2*spl.p))
  x = spl.μ + R*sqrt(spl.Σ)*runifsphere(rng, dim)
  x
end