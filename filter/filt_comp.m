% Comparison of high-pass filters: 3rd-order causal IIR, zero-phase (filtfilt),
% 11-tap linear-phase FIR, and minimum-phase FIR.
% All frequency responses (magnitude, phase, group delay) are MEASURED with
% stepped-sine inputs: a sinusoid is passed through each filter and the output
% amplitude and phase are compared with the input.

fs     = 20000;      % sampling rate (Hz)
fc     = 300;        % high-pass cutoff (Hz)
order1 = 3;          % Butterworth order
thresh = 2e-2;       % settle threshold (fraction of peak)

Wn = fc/(fs/2);                         % cutoff normalized to Nyquist
[d,c] = butter(order1, Wn, 'high');     % 3rd-order Butterworth high-pass

% ---- impulse in a long zero buffer, impulse at the center ----
m   = fs + 1;                           % center index
imp = zeros(2*fs+1,1);  imp(m) = 1;
t   = ((1:length(imp))' - m)/fs;        % time (s), t = 0 at the impulse

h1_causal = filter(d, c, imp);          % 3rd-order causal IIR
h_ff      = filtfilt(d, c, imp);        % zero-phase (forward-backward)

% ---- short linear-phase FIR (11 taps) ----
Ntaps   = 11;
h_short = fir1(Ntaps-1, Wn, 'high');
t_short = (0:numel(h_short)-1)/fs;

% ---- minimum-phase FIR (from an 85-tap linear-phase design) ----
Nlong = 85;
h_lin = fir1(Nlong-1, Wn, 'high');
h_min = firminphase(h_lin);
t_min = (0:numel(h_min)-1)/fs;

% ---- plot colors (one per filter, used in every figure) ----
cIIR = [0.000 0.447 0.741];   % blue
cZP  = [0.850 0.325 0.098];   % red-orange
cFIR = [0.466 0.674 0.188];   % green
cMIN = [0.494 0.184 0.556];   % purple

%% ===== Measured frequency responses (stepped-sine inputs) =====
ftest = logspace(1, log10(0.95*fs/2), 150);  % 10 Hz .. just below Nyquist
Ncyc  = 50;      % cycles per test tone
Lmin  = 2000;    % minimum tone length (samples), >> any filter transient

[g_iir, p_iir, f_iir] = measure_freq_response(@(x) filter(d,c,x),       fs, ftest, Ncyc, Lmin);
[g_ff,  p_ff,  f_ff ] = measure_freq_response(@(x) filtfilt(d,c,x),     fs, ftest, Ncyc, Lmin);
[g_fir, p_fir, f_fir] = measure_freq_response(@(x) filter(h_short,1,x), fs, ftest, Ncyc, Lmin);
[g_min, p_min, f_min] = measure_freq_response(@(x) filter(h_min,1,x),   fs, ftest, Ncyc, Lmin);

% Phase is only defined modulo 360 deg, so choose a reference:
%  - IIR and min-phase FIR: passband (highest frequency) anchored to 0 deg
%  - zero-phase and linear-phase FIR: lowest frequency anchored to its
%    principal value (their phase starts near 0 deg)
p_iir = p_iir - 360*round(p_iir(end)/360);
p_min = p_min - 360*round(p_min(end)/360);
p_ff  = p_ff  - 360*round(p_ff(1)/360);
p_fir = p_fir - 360*round(p_fir(1)/360);

% group delay = -d(phase)/d(omega), in samples
[gd_iir, fmid_iir] = phase_to_gd(p_iir, f_iir, fs);
[gd_ff,  fmid_ff ] = phase_to_gd(p_ff,  f_ff,  fs);
[gd_fir, fmid_fir] = phase_to_gd(p_fir, f_fir, fs);
[gd_min, fmid_min] = phase_to_gd(p_min, f_min, fs);

labels = {'3rd order IIR', 'zero-phase', '11-tap FIR', 'min-phase FIR'};

%% Figure 1: Impulse response
figure;
plot(t, h1_causal, 'Color', cIIR, 'LineWidth', 1.2); hold on;
plot(t, h_ff, '.-', 'Color', cZP, 'LineWidth', 1.2);
plot(t_short, h_short, '--o', 'Color', cFIR, 'LineWidth', 1.2, 'MarkerSize', 3);
plot(t_min,   h_min,   '--',  'Color', cMIN, 'LineWidth', 1.0);
xlim([-0.001 0.0025]);
xline(0, 'k:', 'HandleVisibility', 'off');
legend(labels, 'Location', 'best');
xlabel('Time [s]'); ylabel('Value');
title(sprintf('Impulse response, %d Hz high-pass, Fs = %d', fc, fs)); grid on;

%% Figure 2: Bode magnitude (measured)
figure;
semilogx(f_iir, 20*log10(g_iir), 'Color', cIIR, 'LineWidth', 1.5); hold on;
semilogx(f_ff,  20*log10(g_ff),  'Color', cZP,  'LineWidth', 1.5);
semilogx(f_fir, 20*log10(g_fir), 'Color', cFIR, 'LineWidth', 1.5);
semilogx(f_min, 20*log10(g_min), 'Color', cMIN, 'LineWidth', 1.5);
xline(fc, ':k', 'HandleVisibility', 'off'); yline(-3, ':k', 'HandleVisibility', 'off');
xlim([10 fs/2]); grid on;
title('Bode magnitude');
xlabel('Hz'); ylabel('|H| (dB)');
legend(labels, 'Location', 'best');

%% Figure 3: Bode phase (measured)
figure;
semilogx(f_iir, p_iir, 'Color', cIIR, 'LineWidth', 1.5); hold on;
semilogx(f_ff,  p_ff,  'Color', cZP,  'LineWidth', 1.5);
semilogx(f_fir, p_fir, 'Color', cFIR, 'LineWidth', 1.5);
semilogx(f_min, p_min, 'Color', cMIN, 'LineWidth', 1.5);
xline(fc, ':k', 'HandleVisibility', 'off');
xlim([10 fs/2]); grid on;
title('Bode phase');
xlabel('Hz'); ylabel('phase (deg)');
legend(labels, 'Location', 'best');

%% Figure 4: Group delay (measured)
figure;
plot(fmid_iir, gd_iir, 'Color', cIIR, 'LineWidth', 1.5); hold on;
plot(fmid_ff,  gd_ff,  'Color', cZP,  'LineWidth', 2.0);
plot(fmid_fir, gd_fir, 'Color', cFIR, 'LineWidth', 1.5);
plot(fmid_min, gd_min, 'Color', cMIN, 'LineWidth', 1.5);
xline(fc, ':k', 'HandleVisibility', 'off');
xlim([10 3000]);
ylim([-8 35]);      % below zero so the min-phase FIR and zero-phase line are visible
grid on;
title('Group delay');
xlabel('Hz'); ylabel('delay (samples)');
legend(labels, 'Location', 'best');

%% ===== IIR: poles, zeros, transfer function =====
[z,p,k] = tf2zpk(d, c);

fprintf('\n===== IIR Butterworth, order %d =====\n', order1);
fprintf('Number of poles: %d,  Number of zeros: %d\n', numel(p), numel(z));
fprintf('Zeros (z-plane):\n');   disp(z);
fprintf('  |zeros| = '); fprintf('%.4f ', abs(z)); fprintf('\n');
fprintf('Poles (z-plane):\n');   disp(p);
fprintf('  |poles| = '); fprintf('%.4f ', abs(p)); fprintf(' (stable if <1)\n');
fprintf('Gain k = %.5f\n', k);
fprintf('b (numerator)  : '); fprintf('%.5f ', d); fprintf('\n');
fprintf('a (denominator): '); fprintf('%.5f ', c); fprintf('\n');

if exist('sym', 'file')
    syms z_
    Hz = poly2sym(d, z_) / poly2sym(c, z_);
    fprintf('H(z) = \n'); pretty(vpa(Hz, 5));
end

%% ===== Settling time (samples) =====
% Settle = last sample whose |h| exceeds thresh*peak.
fprintf('\n===== Settling time (thresh = %.0e of peak, fs = %d) =====\n', thresh, fs);

one_sided_settle = @(h, i0) (find(abs(h) > thresh*max(abs(h)), 1, 'last') - i0 + 1);

s3 = one_sided_settle(h1_causal, m);
fprintf('3rd-order IIR (causal)   : %4d samples = %.3f ms (one-sided)\n', s3, s3/fs*1000);

smin = one_sided_settle(h_min, 1);
fprintf('min-phase FIR (causal)   : %4d samples = %.3f ms (one-sided)\n', smin, smin/fs*1000);

sfir = one_sided_settle(h_short, 1);
fprintf('11-tap FIR (causal)      : %4d samples = %.3f ms (one-sided)\n', sfir, sfir/fs*1000);

pk_ff = max(abs(h_ff));
idx   = find(abs(h_ff) > thresh*pk_ff);
pre   = m - idx(1);          % samples before the impulse (lookahead)
post  = idx(end) - m;        % samples after the impulse
tot   = idx(end) - idx(1) + 1;
fprintf('zero-phase (filtfilt)    : total=%4d samples = %.3f ms\n', tot, tot/fs*1000);
fprintf('                           lookahead(pre)=%d, post=%d samples\n', pre, post);

%% ===== Local functions (must stay at the end of the script) =====
function [gain, phase_deg, fmeas] = measure_freq_response(filtfun, fs, ftest, Ncyc, Lmin)
% Stepped-sine measurement of gain and phase.
% filtfun : handle applied to a column vector, e.g. @(x) filter(b,a,x)
% Ncyc    : cycles per test tone
% Lmin    : minimum tone length in samples (must be >> filter transient)
    nf    = numel(ftest);
    gain  = zeros(1, nf);
    ph    = zeros(1, nf);
    fmeas = zeros(1, nf);
    for k = 1:nf
        f0 = ftest(k);
        nc = max(Ncyc, ceil(Lmin*f0/fs));   % enough cycles to reach Lmin samples
        L  = round(nc*fs/f0);
        f0 = nc*fs/L;                       % exact integer number of cycles in L
        n  = (0:L-1)';
        x  = sin(2*pi*f0*n/fs);
        y  = filtfun(x);

        mid = round(L/4)+1 : round(3*L/4);  % middle half, skips the start-up transient
        C   = [cos(2*pi*f0*n(mid)/fs), sin(2*pi*f0*n(mid)/fs)];
        cx  = C \ x(mid);
        cy  = C \ y(mid);
        px  = cx(1) - 1j*cx(2);             % input phasor
        py  = cy(1) - 1j*cy(2);             % output phasor

        gain(k)  = abs(py/px);
        ph(k)    = angle(py/px);
        fmeas(k) = f0;                      % frequency actually tested
    end
    phase_deg = unwrap(ph)*180/pi;
end

function [gd, fmid] = phase_to_gd(phase_deg, f, fs)
% Group delay (samples) from phase (deg) sampled at frequencies f (Hz).
    gd   = -(fs/(2*pi)) * diff(phase_deg*pi/180) ./ diff(f);
    fmid = (f(1:end-1) + f(2:end))/2;
end
