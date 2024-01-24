const ClassID = Symbol

"""
    PCleanNodes
    表示 PCleanClass 依赖图中的节点的抽象类型。
An abstract type representing nodes in a PCleanClass's dependency graph.
"""
abstract type PCleanNode end

"""
    VertexID
    指向 PCleanClass 中 DAG 中一个顶点的整数索引。
An integer index to a vertex in a PCleanClass's DAG.
"""
const VertexID = Int

"""
    AbsoluteVertexID
    通过指定类别 ID 和类中的顶点 ID，绝对顶点 ID 可以明确地标识 PClean 模型中 DAG 节点。

An absolute vertex ID unambiguously identifies a DAG node within a PClean 
model, by specifying a class ID and a vertex ID within the class.
"""
struct AbsoluteVertexID
    class   :: ClassID
    node_id :: VertexID
end

"""
    Path
    表示一系列引用槽的链：class path[i].class 指的是通过其引用槽 path[i].node_id 指向的 class path[i-1].class。路径的最终目标由 path[1].node_id 在 path[1].class 类中的目标隐含确定。

Represents a chain of reference slots: the class path[i].class refers to 
class path[i-1].class via its reference slot path[i].node_id. The ultimate
target of the path is implicitly determined by path[1].node_id's target in 
the path[1].class class.
"""
const Path = Vector{AbsoluteVertexID}



"""
    InjectiveVertexMap
    当类别 A 被类别 B 引用时，类别 B 的 DAG 包括类别 A 的每个节点。InjectiveVertexMap 将类别 A 中的节点 ID 映射到类别 B 中相应节点 ID。类别 A 中类型为 T 的节点将是类别 B 中类型为 SubmodelNode{T} 的节点，或者，如果连接 B 到 A 的槽链超过一个跳跃，则为 SubmodelNode{...SubmodelNode{T}...}。

When class A is referenced by class B, class B's DAG includes
nodes for each of class A's nodes. An InjectiveVertexMap maps
node IDs in class A to their corresponding node IDs in class B.
A node of type T in class A will be a node of type 
SubmodelNode{T} in class B, or, if the slot chain connecting
B to A is longer than one hop, SubmodelNode{...SubmodelNode{T}...}.
"""
const InjectiveVertexMap = Dict{VertexID, VertexID}


mutable struct PitmanYorParams
    strength :: Float64
    discount :: Float64
end

"""
    Step{T}
    `Plan` 树中的节点。`rest` 字段存储树的子节点（计划的“其余”部分），预期为类型 `Plan`。
    类型参数 T 是 Julia 对相互递归类型定义支持不足的一种解决方案。
"""
struct Step{T}
    idx :: VertexID
    rest :: T
end

"""
    Plan
    `Plan` 是具有整数值节点的树的森林。
    这些节点共同覆盖了 PCleanClass 中特定子问题的所有 VertexID。
    给定它们的共同祖先，任何两个节点在条件上是独立的。
"""
struct Plan
    steps :: Vector{Step{Plan}}
end

"""
    PCleanClass

"""
struct PCleanClass
    # 依赖图。
    graph :: DiGraph
  
    # 将顶点编号映射到节点
    nodes :: Vector{PCleanNode}

    # 用于快速查找此类别记录的字段（如果有）。
    # 索引需要更多内存，但如果经常信任某个字段是干净的且已观察到，则可以加速推断。
    hash_keys :: Vector{VertexID}
  
    # 将图节点的子集分区为“块”，对应于 SMC 将依次解决的子问题。
    # 静态节点，即对参数或对其他类别的引用，不在任何块中。
    blocks :: Vector{Vector{VertexID}}

    # 对于每个块，相应的枚举计划
    plans  :: Vector{Plan}

    # 对于每个块，字典将“缺失模式”（观察到的顶点 ID 集）映射到已编译的提议函数。
    # 字典在推断过程中按需填充。
    compiled_proposals :: Vector{Dict{Set{VertexID}, Function}}
  
    # 将用户类别声明中的符号名称映射到计算它们的节点的 ID。
    # 这不仅是调试信息：这些名称是查询和其他类别引用对象属性和引用槽的机制。
    names :: Dict{Symbol, VertexID}
  
    # 每个 incoming_reference 对应于从某个其他 PClean 类别 A 开始并以此类别结束的特定路径。
    # InjectiveVertexMap 将该类别的节点 ID 映射到（可能是间接的）引用类别 A 中的 SubmodelNode ID。
    incoming_references :: Dict{Path, InjectiveVertexMap}
  
    # Pitman-Yor 参数
    initial_pitman_yor_params :: PitmanYorParams
end
  
  

##############
# NODE TYPES #
##############

# 表示确定性计算
struct JuliaNode <: PCleanNode
    f :: Function
    arg_node_ids :: Vector{VertexID}
end

# 表示从原始分布中随机选择的节点
struct RandomChoiceNode <: PCleanNode
    dist :: PCleanDistribution
    arg_node_ids :: Vector{VertexID}
end

# 表示学习参数的声明的节点
struct ParameterNode <: PCleanNode
    make_parameter :: Function
end

# 表示从另一个表中随机选择一行的节点。
# 通过 learned_table_param 指向另一个表。
struct ForeignKeyNode <: PCleanNode
    target_class :: ClassID
    # 将目标类别中的节点 ID 映射到当前类别中的节点 ID，并用于初始化参数值。
    vmap :: InjectiveVertexMap
end

struct SubmodelNode <: PCleanNode
    foreign_key_node_id :: VertexID # 用于查找 gensym
    subnode_id          :: VertexID # 此节点在另一个类别中的 ID；用于在追踪中查找值
    subnode             :: PCleanNode # 其参数根据 *此* 模型的索引进行设置
end

# 表示使用此模型中的值的计算，但在技术上不是此模型的一部分。
struct ExternalLikelihoodNode <: PCleanNode
    path :: Path
    # 引用类别中此节点的 ID。
    external_node_id :: VertexID

    # an ExternalLikelihood node should *only* be a JuliaNode
    # 或随机选择节点（或外键节点，尽管该功能 — DPMem 风格调用类别的特性 — 尚未实现）。
    # Unlike a SubmodelNode's `subnode`, an ExternalLikelihoodNode's `external_node` may reference
    # VertexIDs *not* valid for the current class, but rather the referring class.
    external_node :: Union{JuliaNode, RandomChoiceNode, ForeignKeyNode}
end


# The model itself needn't store the dependency structure,
# I think...
struct PCleanModel
    classes :: Dict{ClassID, PCleanClass}
    class_order :: Vector{ClassID}
end

