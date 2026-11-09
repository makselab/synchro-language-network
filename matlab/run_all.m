%RUN_ALL  Group synchronization matrices and Eq. (1) test for every dataset.
%   Writes ../results/matlab_<dataset>_plv_group.csv and prints the Cluster PLVs.
here = fileparts(mfilename('fullpath'));
results = fullfile(here, '..', 'results');
if ~exist(results, 'dir'), mkdir(results); end
labels = {'SMG_L','SMG_R','PreMA_L','PreMA_R','SMA','WA_L','WA_R','AG_L','AG_R','BA_L','BA_R'};
rest = {{'SMG_L','SMG_R'}, {'PreMA_L','PreMA_R','SMA','WA_L','WA_R'}, {'AG_L','AG_R'}, {'BA_L','BA_R'}};
task = {{'BA_L','SMA'}, {'PreMA_L','PreMA_R'}, {'AG_L','AG_R'}, {'BA_R','WA_L','WA_R'}, {'SMG_L','SMG_R'}};
datasets = {'resting_state_spheres', rest; 'letter_fluency_spheres', task; 'verb_generation_spheres', task; ...
            'resting_state_glasser', rest; 'resting_state_spheres_hrf_deconvolved', rest; ...
            'resting_state_glasser_hrf_deconvolved', rest; 'letter_fluency_spheres_bandpass', task; ...
            'verb_generation_spheres_bandpass', task};
for d = 1:size(datasets, 1)
    name = datasets{d, 1};
    X = load_roi_json(name);
    C = group_plv(X, 5000, 1);
    M = squeeze(mean(C, 1));
    out_file = fullfile(results, ['matlab_' name '_plv_group.csv']);
    if exist('writematrix', 'file')
        writematrix(M, out_file);
    else
        dlmwrite(out_file, M, 'precision', '%.6f');   % GNU Octave
    end
    fprintf('\n%s\n', name);
    out = check_partition(M, datasets{d, 2}, labels);
    for k = 1:numel(out)
        fprintf('  %-32s %.3f   strongest outside link %.3f   Eq. (1) %d\n', ...
                strjoin(out(k).clique, '-'), out(k).cluster_plv, out(k).max_outside, out(k).valid);
    end
end
