function C = surrogate_corrected_plv(X, n_surrogates)
%SURROGATE_CORRECTED_PLV  Surrogate-corrected PLV of one participant (time x ROIs).
%   C(u,v) = (r - g) / (1 - g) if r >= g, 0 otherwise, where r is the PLV
%   between ROIs u and v and g is the mean PLV between ROI u and
%   n_surrogates time-shuffled copies of ROI v (default 5000).
%   Set the random generator with rng() before calling for reproducibility.
if nargin < 2, n_surrogates = 5000; end
[T, R] = size(X);
E = exp(1i * analytic_phase(X));
r = abs(E' * E) / T;
C = zeros(R);
for v = 2:R
    [~, idx] = sort(rand(T, n_surrogates), 1);      % independent permutations
    xv = X(:, v);
    Es = exp(1i * analytic_phase(xv(idx)));          % T x n_surrogates
    for u = 1:v-1
        g = mean(abs(E(:, u)' * Es) / T);
        if r(u, v) >= g
            C(u, v) = (r(u, v) - g) / (1 - g);
            C(v, u) = C(u, v);
        end
    end
end
end
