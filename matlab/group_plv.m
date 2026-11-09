function C = group_plv(X, n_surrogates, seed)
%GROUP_PLV  Surrogate-corrected PLV for every participant.
%   C = GROUP_PLV(X) with X participants x time x ROIs returns
%   participants x ROIs x ROIs; the group matrix is squeeze(mean(C, 1)).
if nargin < 2, n_surrogates = 5000; end
if nargin < 3, seed = 1; end
rng(seed);
[S, ~, R] = size(X);
C = zeros(S, R, R);
for s = 1:S
    C(s, :, :) = surrogate_corrected_plv(squeeze(X(s, :, :)), n_surrogates);
end
end
