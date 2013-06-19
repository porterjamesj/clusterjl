# dumbest possible nearest neighbors search
# computes distance matrix and stores it in memory
# for small data

using Distance

# find all other indicies in matrix M
# for which D is the distance matrix
# that are within r of the point at index i
function nn_search(D::Matrix,
                   i::Int,
                   r::Number)
    neighbors = Int[]
    for j in [1:length(D[:,i])]
        if 0 < D[i,j] < r
            push!(neighbors,j)
        end
    end
    return neighbors
end