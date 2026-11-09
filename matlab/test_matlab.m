%TEST_MATLAB  Checks of the MATLAB functions (also run in GNU Octave).
t = (0:399)' * 2.5;
s1 = sin(2*pi*0.05*t);
% constant phase lag gives PLV close to 1
for lag = [0.3 1 2 3]
    P = plv_matrix([s1, sin(2*pi*0.05*t + lag)]);
    assert(P(1, 2) > 0.98, 'phase lag reduces PLV');
end
% invariance to column order and to a constant offset
rng(2);
X = cumsum(randn(400, 11));
p = randperm(11);
P = plv_matrix(X);
assert(max(max(abs(plv_matrix(X(:, p)) - P(p, p)))) < 1e-10, 'PLV depends on column order');
assert(max(max(abs(plv_matrix(X + 6500) - P))) < 1e-8, 'PLV depends on offset');
% equivalence with the Signal Processing Toolbox, if available
if exist('hilbert', 'file')
    Y = X - mean(X, 1);
    assert(max(max(abs(angle(hilbert(Y)) - analytic_phase(X)))) < 1e-10, 'differs from hilbert');
end
% Eq. (1) with exclusion of more synchronized cliques
M = 0.1 * ones(5); M(1,2) = 0.9; M(2,1) = 0.9;
for pr = [3 4; 3 5; 4 5]', M(pr(1), pr(2)) = 0.6; M(pr(2), pr(1)) = 0.6; end
M(2,3) = 0.7; M(3,2) = 0.7; M(logical(eye(5))) = 0;
out = check_partition(M, {{'a','b'}, {'c','d','e'}}, {'a','b','c','d','e'});
assert(all([out.valid]), 'exclusion rule');
M(1,2) = 0.65; M(2,1) = 0.65;
out = check_partition(M, {{'a','b'}, {'c','d','e'}}, {'a','b','c','d','e'});
assert(~out(1).valid, 'violation not detected');
% agreement with the Python implementation on the resting-state data
ref = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'resting_state_spheres_plv_group.csv');
if exist(ref, 'file')
    Mpy = dlmread(ref, ',', 1, 0);
    C = group_plv(load_roi_json('resting_state_spheres'), 5000, 1);
    d = max(max(abs(squeeze(mean(C, 1)) - Mpy)));
    fprintf('max difference from Python group matrix: %.4f\n', d);
    assert(d < 0.01, 'MATLAB and Python group matrices differ');
end
disp('all MATLAB tests passed');
