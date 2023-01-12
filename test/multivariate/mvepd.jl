using Distributions, Random, LinearAlgebra
using Test

import Distributions: GenericMvEpd
import PDMats: PDMat

@testset "mvepdist" begin
  mu = [1., 2]
  Sigma = [4. 2; 2 3]

  # LogPDF evaluation for varying values of shape parameter
  # Julia's output is compared to R's corresponding values obtained via R's LaplacesDemon package
  # R code exemplifying how the R values (rvalues) were obtained:
  # options(digits=20)
  # library("LaplacesDemon")
  # mu <- 1:2
  # Sigma = matrix(c(4., 2, 2, 3), 2, byrow = T)
  # dmvpe(c(-2, 3), mu, Sigma, 0.5, log = T)

  rvalues = [-10.97517830227108959207, -5.60764219836915422945, -5.29163381793825848831, 
    -5.56509783724926343496, -7.78524200933404575409, -29.07889293348547354867]
  
  
  p = [0.2, 0.5, 0.7, 1, 2, 10]
  for i in eachindex(p)
    d = MvEpd(p[i], mu, Sigma)
    @test isapprox(logpdf(d, [-2., 3]), rvalues[i], atol=1.0e-8)
    dd = typeof(d)(params(d)...)
    @test d.p == dd.p
    @test Vector(d.μ) == Vector(dd.μ)
    @test Matrix(d.Σ) == Matrix(dd.Σ)
  end

  # test constructors for mixed inputs:
  @test typeof(MvEpd(1, Vector{Float32}(mu), PDMat(Sigma))) == typeof(MvEpd(1., mu, PDMat(Sigma)))
  @test typeof(MvEpd(1, mu, PDMat(Array{Float32}(Sigma)))) == typeof(MvEpd(1., mu, PDMat(Sigma)))

  d = GenericMvEpd(1, Array{Float32}(mu), PDMat(Array{Float32}(Sigma)))
  @test convert(GenericMvEpd{Float32}, d) === d
  @test typeof(convert(GenericMvEpd{Float64}, d)) == typeof(GenericMvEpd(1., mu, PDMat(Sigma)))
  @test typeof(convert(GenericMvEpd{Float64}, d.p, d.dim, d.μ, d.Σ)) == typeof(GenericMvEpd(1., mu, PDMat(Sigma)))
  @test partype(d) == Float32
  @test d == deepcopy(d)

  @test length(MvEpd(0.5, mu, Sigma)) == 2
  @test size(rand(MvEpd(.7, mu, Sigma))) == (2,)
  @test size(rand(MvEpd(.7, mu, Sigma), 10)) == (2, 10)

  @test size(rand(MvEpd(0.7, mu, Sigma))) == (2,)
  @test size(rand(MvEpd(0.7, mu, Sigma), 10)) == (2,10)
  @test size(rand(MersenneTwister(123), MvEpd(0.7, mu, Sigma))) == (2,)
  @test size(rand(MersenneTwister(123), MvEpd(0.7, mu, Sigma), 10)) == (2,10)

end