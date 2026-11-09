function P = plv_matrix(X)
%PLV_MATRIX  Phase locking value between all pairs of columns of X (time x ROIs).
E = exp(1i * analytic_phase(X));
P = abs(E' * E) / size(E, 1);
end
