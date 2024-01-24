module PClean
using Distributions
using LightGraphs
using CSV
using DataFrames: DataFrame

include("src/utils.jl")

# Distributions
include("src/distributions/distributions.jl")

# Models
include("src/model/model.jl")
include("src/model/trace.jl")
include("src/model/dependency_tracking.jl")

# DSL
include("src/dsl/builder.jl")
include("src/dsl/syntax.jl")
include("src/dsl/query.jl")

# Inference
include("src/inference/gensym_counter.jl")
include("src/inference/infer_config.jl")
include("src/inference/proposal_row_state.jl")
include("src/inference/block_proposal.jl")
include("src/inference/row_inference.jl")
include("src/inference/inference.jl")
include("src/inference/proposal_compiler.jl")
# include("inference/instrumented_inference.jl")

# Analysis
include("src/analysis.jl")

end # module
