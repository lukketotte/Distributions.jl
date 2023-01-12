using Distributions, Random, StaticArrays, LinearAlgebra
using Test

import Distributions: GenericMvEpd
import PDMats: PDMat

@testset "mvepdist" begin
  mu = [1., 2]
  Sigma = [4. 2; 2 3]

  @test length(MvEpd(0.5, mu, Sigma)) == 2
  @test size(rand(MvEpd(.7, mu, Sigma))) == (2,)
  @test size(rand(MvEpd(.7, mu, Sigma), 10)) == (2, 10)
end