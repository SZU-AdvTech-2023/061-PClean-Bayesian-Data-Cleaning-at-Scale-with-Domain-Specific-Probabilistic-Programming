# 定义一个结构体 AddNoise，表示添加噪声的分布
struct AddNoise <: PCleanDistribution end

# 针对 AddNoise 结构体，指定不使用离散的提议（proposal）
has_discrete_proposal(::AddNoise) = false

# 生成符合正态分布的随机数，参数为均值和标准差
random(::AddNoise, mean::Float64, std::Float64) = rand(Normal(mean, std))

# 计算在给定均值和标准差下，观察到某值的对数概率密度
logdensity(::AddNoise, observed::Float64, mean::Float64, std::Float64) = logpdf(Normal(mean, std), observed)


################
# Learned Mean #
################

# 均值的先验分布结构体 MeanParameterPrior，表示正态分布的均值的先验
struct MeanParameterPrior <: ParameterPrior
  mean :: Float64
  std  :: Float64
end

# 表示正态分布均值的参数结构体 MeanParameter
mutable struct MeanParameter <: BasicParameter
  current_value  :: Float64               # 当前的均值
  prior          :: MeanParameterPrior    # 先验分布
  sample_counts  :: Vector{Int}           # 样本数量
  sample_sums    :: Vector{Float64}       # 样本总和
  sample_stds    :: Vector{Float64}       # 样本标准差
end

# 默认先验，如果用户没有指定，报错提示用户需要指定一个合理的均值参数默认值
default_prior(::Type{MeanParameter}) = begin
  @error "Please specify a reasonable default for the mean parameter."
end

# 用户指定均值的默认先验，可以传入均值 mean 或者均值和标准差 mean、std
default_prior(::Type{MeanParameter}, mean) = MeanParameterPrior(mean, 0.5 * abs(mean))
default_prior(::Type{MeanParameter}, mean, std) = MeanParameterPrior(mean, std)


# 获取参数值的函数
param_value(p::MeanParameter) = p.current_value

# 生成符合 AddNoise 分布的随机数，传入的均值从 MeanParameter 结构体获取
random(a::AddNoise, mean::MeanParameter, std::Float64) = random(a, param_value(mean), std)

# 计算在给定均值和标准差下，观察到某值的对数概率密度，均值从 MeanParameter 结构体获取
logdensity(a::AddNoise, observed::Float64, mean::MeanParameter, std::Float64) = logdensity(a, observed, param_value(mean), std)

# 初始化 MeanParameter 结构体
function initialize_parameter(::Type{MeanParameter}, prior::MeanParameterPrior)
  MeanParameter(rand(Normal(prior.mean, prior.std)), prior, Int[], Float64[], Float64[])
end

# 加入新的观察值，更新样本数量、总和和标准差
function incorporate_choice!(::AddNoise, observed::Float64, mean::MeanParameter, std::Float64)
  idx = findfirst(x -> isapprox(x, std), mean.sample_stds)
  if isnothing(idx)
    push!(mean.sample_stds, std)
    push!(mean.sample_sums, observed)
    push!(mean.sample_counts, 1)
    return
  end
  mean.sample_counts[idx] += 1
  mean.sample_sums[idx] += observed
end

# 移除观察值，更新样本数量、总和和标准差
function unincorporate_choice!(::AddNoise, observed::Float64, mean::MeanParameter, std::Float64)
  idx = findfirst(x -> isapprox(x, std), mean.sample_stds)
  @assert !isnothing(idx)
  mean.sample_counts[idx] -= 1
  mean.sample_sums[idx] -= observed
  if iszero(mean.sample_counts[idx])
    deleteat!(mean.sample_counts, idx)
    deleteat!(mean.sample_sums, idx)
    deleteat!(mean.sample_stds, idx)
  end
end

# Gibbs 抽样更新均值参数
function resample_value!(m::MeanParameter)
  mean, var = m.prior.mean, m.prior.std^2
  for (count, sum, std) in zip(m.sample_counts, m.sample_sums, m.sample_stds)
    # TODO: is this stable?
    new_var = 1.0 / (1.0/var + count/(std^2))
    mean, var = new_var * (mean/var + sum/std^2), new_var
  end
  m.current_value = rand(Normal(mean, sqrt(var)))
end

export AddNoise
