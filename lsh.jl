# locality sensitive hashing for finding nearest neighbors in arrays 
#of n-dimensional vectors

using Distributions
using Distance

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
                       w::Float64,
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





# Find all points within dist of a point in a matrix
# given the matrix and a precompued LSHastable
# d is the distance in which we want to look
# n is the number of neighbors after which to stop looking
# i is the index of the point we want to know about
# M is the matrix
# LSH is the precomputed hashtable
# metric is what distance metric to use
function LS_nearest_neighbors(M::Matrix,LSH::LSHashtable,metric::Metric)
    # compute hashes with each function for the 
    point_hashes = [func(M[:,i]) for func in LSH.functions]

    # iterate over the correct bucket in each hash table and look for neighbors
    
end