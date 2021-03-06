# locality sensitive hashing for finding nearest neighbors in arrays 
#of n-dimensional vectors

using Distributions
using Distance

require("dumbsearch.jl")

type LSHashtable
    functions::Array{Function}
    tables::Array{Dict{Uint,Set{Int}}}
end

# Construct a new hash function g, composed of k dot products
# for vectors of length d, with bandwidth w
function ls_hash(d::Int,w::Float64,k::Int)
    hashfuncs = Function[]
    for i=[1:k]
        projvec = rand(Normal(0,1),d)
        b = rand(Uniform(0,w))
        push!(hashfuncs,v -> floor((dot(v,projvec)+b)/w))
    end
    return v -> hash([hashfuncs[i](v) for i in [1:k]])
end

# Compute locality sensitive hash tables for a matrix.
# M is the matrix, should be shaped [dimensions,datapoints]
# L is the number of tables you want, w is the bandwidth,
# k is number of dot products to use in constructing each hash function.
function ls_hashtables(M::Matrix,
                       L::Int,
                       w::Number,
                       k::Int)
    # facts about the Matrix
    d = length(M[:,1]) # dimensionality
    l = length(M[1,:]) # number of data points

    # generate the hash functions
    hashfuncs = [ls_hash(d,w,k) for i in [1:L]]

    # generate the tables
    tables = [Dict{Uint,Set{Int}}() for i in [1:L]]

    # populate the table by hashing each datapoint
    # with each of the L generated hash functions
    for i in [1:l] # i is index into matrix
        for j in [1:L] # j is index into array of hashtables
            this_hash = hashfuncs[j](M[:,i]) 
            if haskey(tables[j],this_hash) == false
                tables[j][this_hash] = Set{Int}(i)
            else
                add!(tables[j][this_hash],i)
            end
        end
    end
    return LSHashtable(hashfuncs,tables)
end


# Find points within a given distances of a point in a matrix
# given the matrix and a precompued LSHastable.
# d is the distance away from the query point we want to look
# pass Inf if you want the n nearest neighbors regardless of distance
# n is the number of neighbors desired
# q is the index of the query point
# M is the matrix
# LSH is the precomputed hashtable
# metric is what distance metric to use
function ls_nearest_neighbors(M::Matrix,
                              LSH::LSHashtable,
                              metric::Metric,
                              q::Int,
                              n::Number,
                              d::Number)
    # dimensionality of the data
    dims = length(M[:,1])
    
    # compute the hash of the query point with each function 
    query_hashes = [func(M[:,q]) for (i,func) in enumerate(LSH.functions)]
    
    # iterate over each hash table and collect neighbors
    neighbors = Set{Int}()
    for (i,table) in enumerate(LSH.tables)
        if haskey(table,query_hashes[i])
            for neighbor in table[query_hashes[i]]
                this_dist = pairwise(metric,reshape(M[:,neighbor],(dims,1)),
                                            reshape(M[:,q],(dims,1)))
                if this_dist[1] < d && neighbor != q
                    add!(neighbors,neighbor)
                    if length(neighbors) == n
                        return neighbors
                    end
                end
            end
        end
    end
    # if we get here, we didn't find enough neighbors,
    # so just return the ones we did
    return neighbors
end

# test for correctness
test_cor(i,dist) = nn_search(D,i,dist) == 
sort([i for i in ls_nearest_neighbors(A,lsh,Euclidean(),i,Inf,dist)])

# test speed

function test_nn(A)
    D = pairwise(Euclidean(),A)
    for i in 0:100
        nn_search(D,4,5)
    end
end

function test_lsh(A)
    lsh = ls_hashtables(A,50,4.0,13)
    for i in 0:100
        ls_nearest_neighbors(A,lsh,Euclidean(),12,Inf,6)
    end
end

A = rand(30,10000)
@elapsed test_nn(A)
@elapsed test_lsh(A)

# TODO: - implement parameter optimization for a given data set
#       - change API?