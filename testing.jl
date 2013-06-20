function gen_clustered_data(;nclusters = 3, n_features = 2,
                            n_samples_per_cluster = 20, std = 0.4)
    means = [ 1 -1  1 -1 
              1 -1 -1  1
              1  0  1  1
              0  1  1  0] + 10
    
    X = Array(Int,(n_features,0))
    for ncluster in 1:nclusters
        X = cat(2,X,
                bsxfun(+,means[1:n_features,ncluster],
                       rand(n_features,
                            n_samples_per_cluster)))
    end
    return X
end