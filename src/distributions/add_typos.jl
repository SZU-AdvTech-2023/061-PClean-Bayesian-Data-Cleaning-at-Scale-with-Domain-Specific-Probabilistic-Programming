# 导入 StringDistances 库中的 DamerauLevenshtein 和 evaluate 函数
import StringDistances: DamerauLevenshtein, evaluate

# 定义结构体 AddTypos，表示添加打字错误的分布
struct AddTypos <: PCleanDistribution end

# 针对 AddTypos 结构体，指定不使用离散的提议（proposal）
has_discrete_proposal(::AddTypos) = false

# 支持明确缺失的观察值
supports_explicitly_missing_observations(::AddTypos) = true

# 执行打字错误，包括插入、删除、替换和转置
perform_typo(typo, word) = begin
  # 如果是插入
  if typo == :insert
    index = rand(DiscreteUniform(0, length(word)))
    letter = collect('a':'z')[rand(DiscreteUniform(1, 26))]
    return "$(word[1:index])$(letter)$(word[index+1:end])"
  end
  if typo == :delete
    index = rand(DiscreteUniform(1, length(word)))
    return "$(word[1:index-1])$(word[index+1:end])"
  end
  if typo == :substitute
    index = rand(DiscreteUniform(1, length(word)))
    letter = collect('a':'z')[rand(DiscreteUniform(1, 26))]
    return "$(word[1:index-1])$letter$(word[index+1:end])"
  end
  if typo == :transpose
    if length(word) == 1
      return
    end
    index = rand(DiscreteUniform(1, length(word)-1))
    return "$(word[1:index-1])$(word[index+1])$(word[index])$(word[index+2:end])"
  end
end

# 常量，表示不可能的概率
const IMPOSSIBLE = -1e5

# 随机生成带有打字错误的单词
random(::AddTypos, word::String, max_typos=nothing) = begin
  num_typos = rand(NegativeBinomial(ceil(length(word) / 5.0), 0.9))
  num_typos = isnothing(max_typos) ? num_typos : min(max_typos, num_typos)
  for i=1:num_typos
    # Randomly insert/delete/transpose/substitute
    typo = [:insert, :delete, :transpose, :substitute][rand(DiscreteUniform(1, 4))]
    word = perform_typo(typo, word)
  end
  return word
end

# 常量，存储添加打字错误的概率字典
const add_typos_density_dict = Dict{Tuple{String, String}, Float64}()
# 每个打字错误的平均字符数
const LETTERS_PER_TYPO = 5.0

# 计算在给定观察值和单词下的对数概率密度
logdensity(::AddTypos, observed::Union{String,Missing}, word::String, max_typos=nothing) = begin
  # 如果观察值为缺失值，则返回概率密度为0
  if ismissing(observed)
    return 0.0
  end

  # 使用 DamerauLevenshtein 计算观察值和单词之间的编辑距离
  get!(add_typos_density_dict, (observed, word)) do
    num_typos = evaluate(DamerauLevenshtein(), observed, word)
    # 如果设置了最大打字错误数，并且实际打字错误数超过了最大值，则返回不可能的概率
    if !isnothing(max_typos) && num_typos > max_typos
      return IMPOSSIBLE
    end

    # 计算打字错误概率的对数
    l = logpdf(NegativeBinomial(ceil(length(word) / LETTERS_PER_TYPO), 0.9), num_typos)
    l -= log(length(word)) * num_typos
    l -= log(26) * (num_typos) / 2 # 可能应该实际计算最可能的打字错误路径的概率
    l
  end
end

export AddTypos
