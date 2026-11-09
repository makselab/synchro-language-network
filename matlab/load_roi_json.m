function [X, meta] = load_roi_json(name, data_dir)
%LOAD_ROI_JSON  Read data/<name>.json.
%   [X, meta] = LOAD_ROI_JSON(name) returns X (participants x time points x ROIs)
%   and the metadata. ROI order: SMG_L SMG_R PreMA_L PreMA_R SMA WA_L WA_R
%   AG_L AG_R BA_L BA_R.
if nargin < 2
    data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
end
meta = jsondecode(fileread(fullfile(data_dir, [name '.json'])));
X = double(meta.data);
meta = rmfield(meta, 'data');
expected = [meta.n_participants, meta.n_timepoints, meta.n_rois];
if ~isequal(size(X), expected)
    error('load_roi_json:shape', '%s: data size %s differs from metadata %s', ...
          name, mat2str(size(X)), mat2str(expected));
end
if any(~isfinite(X(:)))
    error('load_roi_json:values', '%s: non-finite values', name);
end
end
