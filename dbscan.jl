using Devectorize
using Distance
# using MLBase

require("dumbsearch.jl")

# This follows wikipedia's pseudocode

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


# this follows what scikit does
# The array is treated as a feature array,
# unless metric is given as None, in which case it is assumed
# that X is a precomputed distance matrix 
function dbscan2(X::Matrix,
                 eps::Number,
                 min_pnts::Int,
                 metric::Union(Metric,String))
    # number of features
    n = size(X)[2]
    # order to iterate over indices
    # I feel like there should be a way to do this without calculating
    # a whole new vector (e.g. like an iterator in python)
    index_order = randperm(n)
    # get distance matrix
    if metric != "precomputed"
        D = pairwise(metric,X)
    else
        D = X
    end
    # Calculate the epsilon neighborhood for each sample
    # Note that this marks each sample as a member of its own
    # epsilon neighborhood.
    # This smells bad, I bet there's a better way
    neighborhoods = [filter(x -> D[x,i] < eps,[1:n]) for i=[1:n]]
    # Initially, all samples are noise (labeled -1)
    labels = fill(-1,n)
    # Keep track of each core sample
    core_samples = Int[]
    # the label for the new cluster
    label_num = 0
    # Iterate over samples in a random order,
    # deciding if each is a core sample and if so
    # build a cluster from it
    for i in index_order
        if labels[i] != -1 || length(neighborhoods[i]) < min_pnts
            # this point is already classified or is not a core point
            continue
        end
        # this is a core sample
        push!(core_samples,i)
        labels[i] = label_num
        # expand the cluster
        candidates = [i]
        while length(candidates) > 0
            new_candidates = Int[]
            # A candidate is a core point in the current cluster that
            # has not yes been used for expansion
            for c in candidates
                # get the indicies of all currently noise points in this
                # candidate's epsilon neighborhood and make them
                # part of this cluster
                noise = filter(x -> labels[x] == -1, neighborhoods[c])
                labels[noise] = label_num
                for neighbor in noise
                    # check if this neighbor is also a core point
                    if length(neighborhoods[neighbor]) >= min_pnts
                        # it is a core point, add it to the new candidates
                        push!(new_candidates,neighbor)
                        push!(core_samples,neighbor)
                    end
                end
            end
            candidates = new_candidates
        end
        # Done making current clutser
        # Next found core point starts a new cluster
        label_num += 1
    end
    return core_samples, labels
end