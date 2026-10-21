% Filter a single-channel snippet two ways and compare, around a known spike.
%   symmetric : zero-phase filtfilt  (== kilosort / scipy filtfilt)
%   causal    : 3rd-order Butterworth high-pass, one-directional (filter)
%
% Produces TWO figures, each ONE figure with TWO stacked panels:
%   top panel = raw (original), bottom panel = both filtered traces.
%   Figure 1 : 14 ms window around the spike
%   Figure 2 : one spike, +/- n_side samples around the peak
%
% Data: cropped_ch4.csv = one column, int16 values, channel 4 (kilosort 0-based).
% NOTE: crop_start MUST match how cropped_ch4.csv was cut from the .dat.

%% ----- parameters -----
fs    = 20000;      % sampling rate (Hz)
fc    = 300;        % high-pass cutoff (Hz)
order = 3;          % 3rd-order Butterworth

% spike of interest (convert kilosort 0-based -> MATLAB 1-based)
spike_global = 2657;          % global sample index (0-based, kilosort)
crop_start   = 2600;          % start of the crop  <-- must match the CSV's crop!
spike_idx    = spike_global - crop_start + 1;   % MATLAB 1-based sample

main_ch0 = 4;                 % kilosort main channel (0-based, for labels only)
templ    = 10;                % template id (for the title only)

fig1_ms  = 14;                % Figure 1: window width in ms
n_side   = 20;                % Figure 2: samples before and after the peak

%% ----- load one channel -----
x = readmatrix('cropped_ch4.csv');     % column vector
x = x(:);
nsamp = numel(x);
fprintf('Loaded %d samples, 1 channel (%.2f ms)\n', nsamp, nsamp/fs*1e3);

%% ----- design filter (same as Python butter(3, 300/(fs/2), 'high')) -----
Wn = fc/(fs/2);                    % cutoff normalized to Nyquist
[b,a] = butter(order, Wn, 'high');

%% ----- apply along time -----
y_sym    = filtfilt(b, a, x);      % symmetric / zero-phase
y_causal = filter(b, a, x);        % causal 3rd-order IIR

%% ----- x axis (in samples) -----
samp    = crop_start + (0:nsamp-1);        % global sample number per point
spike_x = spike_global;                    % spike position on the sample axis

%% ----- window index ranges -----
% Figure 1: a fixed 14 ms window centred on the spike, shifted to stay in range
half  = round((fig1_ms/1000)*fs/2);        % samples each side (14 ms -> 140)
lo = spike_idx - half;  hi = spike_idx + half;
if lo < 1,      lo = 1;      hi = min(nsamp, lo + 2*half); end
if hi > nsamp,  hi = nsamp;  lo = max(1,     hi - 2*half); end
win14 = lo:hi;

% Figure 2: single spike, +/- n_side samples
win1 = max(1, spike_idx-n_side) : min(nsamp, spike_idx+n_side);

%% ----- Figure 1: 14 ms window -----
plot_two_panel(samp, x, y_sym, y_causal, win14, spike_x, spike_global, ...
    sprintf('%d Hz highpass, ch %d (template %d) - %g ms window', ...
            fc, main_ch0, templ, fig1_ms));

%% ----- Figure 2: single spike, +/- n_side samples -----
plot_two_panel(samp, x, y_sym, y_causal, win1, spike_x, spike_global, ...
    sprintf('%d Hz highpass, ch %d (template %d) - single spike (+/-%d samp)', ...
            fc, main_ch0, templ, n_side));


%% ========================================================================
function plot_two_panel(xax, x, y_sym, y_causal, win, spike_x, spike_global, ttl)
% One figure, two stacked panels: raw on top, both filtered traces on bottom.
% xlim is set to the window; ylim is fitted to the data inside the window.

    xl = [xax(win(1)), xax(win(end))];

    % tight y-limits with a small margin
    pad_frac = 0.08;
    rraw = [min(x(win)),  max(x(win))];
    rflt = [min([y_sym(win); y_causal(win)]), max([y_sym(win); y_causal(win)])];
    draw = diff(rraw)*pad_frac + eps;   dflt = diff(rflt)*pad_frac + eps;
    ylraw = rraw + [-draw draw];        ylflt = rflt + [-dflt dflt];

    figure;

    % --- top: original ---
    ax1 = subplot(2,1,1);
    plot(xax, x, 'Color', [.5 .5 .5], 'LineWidth', 1.2); hold on;
    xline(spike_x, 'k:', sprintf('spike @ %d', spike_global), ...
          'LabelVerticalAlignment','top');
    yline(0, 'k-', 'Alpha', 0.3);
    grid on; xlim(xl); ylim(ylraw);
    ylabel('Amplitude'); title(ttl);
    legend('raw (original)', 'Location', 'best');

    % --- bottom: filtered ---
    ax2 = subplot(2,1,2);
    plot(xax, y_sym,    'LineWidth', 1.5); hold on;
    plot(xax, y_causal, 'LineWidth', 1.5);
    xline(spike_x, 'k:');
    yline(0, 'k-', 'Alpha', 0.3);
    grid on; xlim(xl); ylim(ylflt);
    xlabel('Sample'); ylabel('Amplitude');
    legend('symmetric (filtfilt)', 'causal 3rd-order IIR', 'Location', 'best');

    linkaxes([ax1 ax2], 'x');
end