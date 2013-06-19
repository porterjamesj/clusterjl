using Devectorize
using Distance
using MLBase

# Find k points closest to x. This excludes x
function k_nearest_neighbors(k::Integer, x::Integer, D::Matrix)
    n = size(D, 2)
    ranked_by_proximity = sortperm(reshape(D[x, :], n))
    res = Array(Int, k)
    index, filled = 1, 1
    while filled <= k
        if ranked_by_proximity[index] == x
            index += 1
            continue
        else
            res[filled] = ranked_by_proximity[index]
            filled += 1
            index += 1
        end
    end
    return res
end

function(data::Array,eps::Number,minPnts::Int,metric::Metric)
    unvisited = trues(size(data)[1]) # keep track of which points are still unvisited
    clusters = [] # array of clusters, each of which is itself a 1-d array
    
    distmat = pairwise(metric,data)

    #iterate until all points have been visited
    while any(unvisited)
        curi = findfirst(unvisited)
        unvisited[curi] = false
        neighbors = k_nearest_neighbors