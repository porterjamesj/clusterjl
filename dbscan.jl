using Devectorize
using Distance
# using MLBase

require("dumbsearch.jl")

function expand_cluster(p::Int,
                        neighbors::Array{Int})
    cluster = []
    push!(cluster, p)
    incluster(p) = true
    for p_ in neighbors
        if unvisited(p_)
            unvisited(p_) = false 
            neighbors_ = nn_search(distmat, eps, p_)
            if length(neighbors_) >= min_pnts
                append!(neighbors,neighbors_)
            end
        end
        if !incluster(p_)
            push!(cluster,p_)
        end
    end
    return cluster
end

function db_scan(data::Array,
                 eps::Number,
                 min_pnts::Int,
                 metric::Metric)
    unvisited = trues(size(data)[1]) # keep track of which points are still unvisited
    noise = falses(size(data)[1]) # which points are noise
    incluster = copy(noise) # which points have already been added to a cluster
    clusters = [] # array of clusters, each of which is itself a 1-d array
    
    distmat = pairwise(metric,data)
    
    #iterate until all points have been visited
    while any(unvisited)
        curi = findfirst(unvisited)
        unvisited[curi] = false
        neighbors = nn_search(distmat, eps, curi)
        if length(neighbors) < min_pnts
            curcluster = []
            push!(clusters,expand_cluster(curi,neighbors))
        end
    end
    return clusters
end