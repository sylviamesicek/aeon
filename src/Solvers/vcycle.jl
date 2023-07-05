

function vcycle!(x, A, b, prolong, restrict)
    solver = IterativeSolvers.gmres_iterable!(x, A, b; maxiter=10)

    # Presmooth
    for _ in solver; end

    # Compute residual
    residual = solver.residual.current

    # Restrict
    resdidual_H = restrict * residual

    error_H = zeros(length(resdidual_H))

    #vcycle!(error_H, A, B, prolong, restrict)

    residual.x += prolong * error_H

    # Postsmoothing
    for _ in solver; end
end