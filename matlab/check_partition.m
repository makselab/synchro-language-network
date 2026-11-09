function out = check_partition(M, partition, labels)
%CHECK_PARTITION  Test Eq. (1) for each clique of a partition.
%   out = CHECK_PARTITION(M, partition, labels): M is the group synchronization
%   matrix, partition a cell array of cell arrays of ROI labels, labels the ROI
%   labels of M. Cliques are taken in decreasing order of Cluster PLV; each is
%   compared with the ROIs that do not belong to it or to a clique with higher
%   Cluster PLV. out is a struct array with fields clique, cluster_plv,
%   max_outside, max_outside_link, valid.
n = numel(partition);
idx = cell(1, n);
value = zeros(1, n);
for c = 1:n
    [ok, idx{c}] = ismember(partition{c}, labels);
    if ~all(ok), error('check_partition:labels', 'unknown ROI label'); end
    pairs = nchoosek(idx{c}, 2);
    value(c) = mean(M(sub2ind(size(M), pairs(:, 1), pairs(:, 2))));
end
allidx = [idx{:}];
if numel(unique(allidx)) ~= numel(allidx), error('check_partition:overlap', 'cliques overlap'); end
[~, order] = sort(value, 'descend');
assigned = [];
out = struct('clique', {}, 'cluster_plv', {}, 'max_outside', {}, 'max_outside_link', {}, 'valid', {});
for c = order
    others = setdiff(1:size(M, 1), [idx{c}, assigned]);
    best = 0; link = {'', ''};
    for k = idx{c}
        for o = others
            if M(k, o) > best, best = M(k, o); link = {labels{k}, labels{o}}; end
        end
    end
    out(end + 1) = struct('clique', {partition{c}}, 'cluster_plv', value(c), ...
                          'max_outside', best, 'max_outside_link', {link}, 'valid', value(c) >= best); %#ok<AGROW>
    assigned = [assigned, idx{c}]; %#ok<AGROW>
end
end
