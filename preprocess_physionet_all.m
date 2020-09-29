clear all

% DEFINE GLOBALS
DATA_PATH = 'E:\Users\Aaron\PhysioNet';
SAVE_DIR  = '.\data\CAP\';
FS        = 100;

files = dir(DATA_PATH);
files = files(3:end); % remove '.' and '..'

for f = 1:length(files)
    %% Get data
    filename          = files(f).name;
    [header, datamat] = edfread(fullfile(DATA_PATH, filename)); % parse data from EDF file
    channel           = find(contains(header.label, {'ECG', 'EKG', 'ekg'}));    % get ECG channel
    if isempty(channel)
        header.label
        filename
        continue
    end
    ECG               = datamat(channel,:);                     % get ECG signal
    originalFs        = header.frequency(channel);              % get recorded sampling rate

    %% Pre-process
    % Decimate
    ECG = resample(ECG, FS, originalFs);

    % Comb filter
    Q      = 50;
    F0     = 50;
    bw     = (F0/(FS/2))/Q;
    w      = round(FS/F0);
    [b, a] = iircomb(w, bw, 'notch');
    ECG    = filtfilt(b, a, double(ECG));

    % Bandpass filter
    cutoff = [5 25];
    Fn     = cutoff/(FS/2); % Normalized cutoff frequency/ies
    ord    = 2;             % Filter order
    [b, a] = butter(ord, Fn, 'bandpass');
    ECG    = filtfilt(b, a, double(ECG));

    %% Find QRS true

    pkheight = prctile(ECG,96);
    [~, idx] = findpeaks(-ECG, 'minpeakheight', pkheight, 'minpeakprominence', pkheight, 'minpeakdistance', FS/2);

    y = zeros(size(ECG));
    QRS_length = round(FS/10);
    for n = 1:length(idx)
        idx1 = max(idx(n) - QRS_length/2, 1);
        idx2 = min(idx(n) + QRS_length/2, length(ECG));
        y(idx1:idx2) = 1;
    end

    %% Save to disk
    [~, name]  = fileparts(files(f).name);
    prefix     = sprintf('%s_ECG_Fs%dHz_', name, FS); % e.g. 'n1_ECG_Fs100Hz_'
    save(fullfile(SAVE_DIR, 'signals',    [prefix 'BP5-25Hz.mat']),    'ECG');
    save(fullfile(SAVE_DIR, 'labels_idx', [prefix 'beatindices.mat']), 'idx');
    save(fullfile(SAVE_DIR, 'labels_seq', [prefix 'labels.mat']),      'y');
end
