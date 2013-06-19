using Devectorize
using Distance
using MLBase

# type to represent a node in the generated tree
type HNode
    left::Union(HNode,Int) # the int is an index into the array of nodes
    right::Union(HNode,Int)
    size::Int
end

function hcluster(points::Array,dist::Metric)
    # first calculate pairwise distances using the specified distance
    distmatrix = pairwise(dist,points)
    tril(distmatrix)
    # now find the closest pair of points and merge them
    point,idx = findmin(filter(v -> v > 0,distmatrix))
    tomerge = ind2sub(size(distmatrix),idx-1) #indices of the clusters to merge


    #merge them
    
    
end