############################
## Exports #################
############################

export WLSMesh, meshmatrix

############################
## Standard ################
############################

struct WLSMesh{Dim, Field, SCount, BCount, BFunc, WFunc} 
    domain::Domain{Dim, Field}
    supports::Matrix{Int}
    distances::Matrix{Field}
    weight::WFunc
    basis::Basis{BCount, BFunc}

    function WLSMesh{SCount}(domain::Domain{Dim, Field}, basis::Basis{BCount, BFunc}, weight::WFunc) where {Dim, Field, BCount, SCount, BFunc <: AnalyticFunction, WFunc <: AnalyticFunction}
        if SCount < 2
            error("SCount is $(SCount). For Weighted Least Squares algorithm, scount must be at least 2")
        end

        points = position_array(domain)
        kdtree = KDTree(points)

        idxs, dists = knn(kdtree, points, SCount)

        supports = reduce(hcat, idxs)
        distances = reduce(hcat, dists)

        new{Dim, Field, SCount, BCount, BFunc, WFunc}(domain, supports, distances, weight, basis)
    end
end

function meshmatrix(mesh::WLSMesh{Dim, Field, SCount}) where {Dim, Field, SCount}
    rowindices = Vector(undef, SCount * length(mesh.domain))
    colindices = Vector(undef, SCount * length(mesh.domain))
    values = zeros(SCount * length(mesh.domain))

    for pidx in eachindex(mesh.domain)
        offset = SCount * (pidx - 1)
        for sidx in 1:SCount
            rowindices[offset + sidx] = pidx
            colindices[offset + sidx] = mesh.supports[sidx, pidx]
        end
    end

    sparse(rowindices, colindices, values)
end