function phi = analytic_phase(X)
%ANALYTIC_PHASE  Instantaneous phase of each column of X (time x ROIs).
%   The analytic signal is computed along time (dimension 1), separately for
%   each ROI, after centring each series; the phase is the full angle in
%   (-pi, pi]. Equivalent to angle(hilbert(X - mean(X))) of the Signal
%   Processing Toolbox, without requiring it.
X = X - mean(X, 1);
N = size(X, 1);
h = zeros(N, 1);
if mod(N, 2) == 0
    h([1, N/2 + 1]) = 1;
    h(2:N/2) = 2;
else
    h(1) = 1;
    h(2:(N + 1)/2) = 2;
end
z = ifft(fft(X, [], 1) .* h, [], 1);
phi = angle(z);
end
