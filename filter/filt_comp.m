% Zero-phase (filtfilt) vs causal Butterworth high-pass: impulse response
fs     = 20000;      % sampling rate (Hz)
fc     = 300;        % high-pass cutoff (Hz)
order  = 2;          % Butterworth order
order1 = 3;
thresh = 2*1e-2;       % settle threshold (fraction of peak)

Wn = fc/(fs/2);                        % cutoff normalized to Nyquist (MATLAB convention)
%[b,a] = butter(order, Wn, 'high');     % 2rd-order Butterworth high-pass
[d,c] = butter(order1, Wn, 'high');     % 3rd-order Butterworth high-pass
m  = fs + 1;                         % center index
imp  = zeros(2*fs+1,1);  imp(m) = 1;           % unit impulse at the CENTER

%causal 2nd and 3rd order and min-phase filters
%h_causal = filter(b, a, imp);      % one-directional (causal) response
h1_casual = filter(d, c, imp);
h_ff = filtfilt(d, c, imp);   % zero-phase (kilosort, == scipy filtfilt)
t = ((1:length(imp)) - m)/fs;     % samples relative to impulse (0 = impulse)

% ---- Short linear-phase FIR (11 taps) ----
Ntaps = 11;
h_short = fir1(Ntaps-1, Wn, 'high');      % linear-phase FIR high-pass, 11 taps
t_short = (0:numel(h_short)-1) / fs;   % 11 taps, centered

% ---- Minimum-phase FIR (same magnitude as a longer FIR) ----
Nlong = 85;
h_lin = fir1(Nlong-1, Wn, 'high');
% Convert to minimum phase
h_min = firminphase(h_lin);  
t_min = (0:numel(h_min)-1) / fs;            % 41 taps, starts at 0

% calculating the Magnitude and Phase of filtfilt by sinusoid inputs 
ftest = logspace(1, log10(fs/2), 40);   % 40 test frequencies, 10 Hz .. Nyquist
gain_ff  = zeros(size(ftest));
phase_ff = zeros(size(ftest));

for k = 1:numel(ftest)
    f0 = ftest(k);
    Ncyc = 50;
    L  = round(Ncyc*fs/f0);   % pick a whole number of samples (approx 50 cycles)
    f0 = Ncyc*fs/L;           % now RECOMPUTE f0 so it's EXACTLY Ncyc cycles in L samples
    n  = (0:L-1);         % 50 cycles of this frequency
    x  = sin(2*pi*f0*n/fs);              % input sine
    y  = filtfilt(d, c, x);              % output

    mid = round(numel(n)/4) : round(3*numel(n)/4);   % keep middle half only
    nn = n(mid);
    C  = [cos(2*pi*f0*nn/fs).', sin(2*pi*f0*nn/fs).'];
    cx = C \ x(mid).';        % fit input   -> [a; b]
    cy = C \ y(mid).';        % fit output
    px = cx(1) - 1j*cx(2);    % input phasor
    py = cy(1) - 1j*cy(2);    % output phasor
    gain_ff(k)  = abs(py)/abs(px);
    phase_ff(k) = angle(py/px)*180/pi;
end

%filtfilt group delay
phi = unwrap(phase_ff*pi/180); % back to radians, unwrapped
gd_meas = -(fs/(2*pi)) * diff(phi)./diff(ftest);   % group delay in samples
fmid    = (ftest(1:end-1)+ftest(2:end))/2;         % midpoint frequencies

% --- frequency response ---
% 2nd order causal
%nfft = 4096;
%[H, f]   = freqz(b, a, nfft, fs);          % single-pass complex response
%mag_db  = 20*log10(abs(H));               % filter (causal) magnitude
%phase   = unwrap(angle(H))*180/pi;        % filter phase
%phase = phase - 360*round(phase(end)/360);
%[gd, fg] = grpdelay(b, a, nfft, fs);       % filter group delay (samples)

% 3rd order causal
nfft = 4096;
[H1, f1]   = freqz(d, c, nfft, fs);          % single-pass complex response
mag1_db  = 20*log10(abs(H1));               % filter (causal) magnitude
phase1   = unwrap(angle(H1))*180/pi;        % filter phase
phase1 = phase1 - 360*round(phase1(end)/360);
[gd1, fg1] = grpdelay(d, c, nfft, fs);       % filter group delay (samples)

%FIR frequency responses
[Hf, ff] = freqz(h_short, 1, nfft, fs);     % 11-tap FIR
[Hm, fm] = freqz(h_min, 1, nfft, fs);     % min-phase FIR
magf_db = 20*log10(abs(Hf));
magm_db = 20*log10(abs(Hm));
phasef = unwrap(angle(Hf))*180/pi;
phasef = phasef - 360*round(phasef(end)/360);
phasem = unwrap(angle(Hm))*180/pi;
phasem = phasem - 360*round(phasem(end)/360);
[gdf, fgf] = grpdelay(h_short, 1, nfft, fs);
[gdm, fgm] = grpdelay(h_min, 1, nfft, fs);

%% Figure 1: Impulse Response
figure;
%plot(t, h_causal, 'LineWidth', 1.2); hold on;
plot(t, h1_casual, 'LineWidth', 1.2); hold on;
plot(t, h_ff, '.-', 'LineWidth', 1.2);
plot(t_short, h_short, '--', 'LineWidth',1.2);
plot(t_min, h_min, '--', 'LineWidth',1.0);
xlim( [ -0.001 0.0025 ] );
xline(0,'k:');
legend('3rd order IIR', 'zero-phase', '11-tap FIR', 'min-phase FIR', 'Location','best');
xlabel( 'Time [s]' ), ylabel( 'Value' );
title(sprintf( 'Impulse Response, %d Hz highpass, Fs = %d', fc, fs)); grid on;

%% Figure 2: Bode magnitude (both)
figure;
%semilogx(f, mag_db,'LineWidth',1.5); hold on;
semilogx(f1, mag1_db,'LineWidth',1.5); hold on;
semilogx(ftest, 20*log10(gain_ff), '.-'); grid on;
semilogx(ff, magf_db, 'LineWidth', 1.5);
semilogx(fm, magm_db, 'LineWidth', 1.5);
xline(fc,':k'); yline(-3,':k');
xlim([10 fs/2]); grid on;
title('Bode magnitude');
xlabel('Hz'); ylabel('|H| (dB)');
legend('3rd order IIR', 'zero-phase', '11-tap FIR', 'min-phase FIR', 'Location','best');

%% Figure 3: Bode phase (both)
figure;
%semilogx(f, phase,'LineWidth',1.5); hold on;
semilogx(f1, phase1,'LineWidth',1.5); hold on;
semilogx(ftest, phase_ff, '.-'); grid on;
semilogx(ff, phasef, 'LineWidth', 1.5);
semilogx(fm, phasem, 'LineWidth', 1.5);
xline(fc,':k');
xlim([10 fs/2]); grid on;
title('Bode phase -');
xlabel('Hz'); ylabel('phase (deg)');
legend('3rd order IIR', 'zero-phase', '11-tap FIR', 'min-phase FIR', 'Location','best');

%% --- group delay vs frequency (the phase nonlinearity) ---
figure;
%plot(fg, gd, 'LineWidth',1.5); hold on;
plot(fg1, gd1, 'LineWidth', 1.5); hold on;
plot(fmid, gd_meas);
plot(fgf, gdf, 'LineWidth', 1.5);
plot(fgm, gdm, 'LineWidth', 1.5);
xlim([10 3000]);
ylim([0 35]);  
grid on;
title('Group delay');
xlabel('Hz'); ylabel('delay (samples)');
legend('3rd order IIR', 'zero-phase', '11-tap FIR', 'min-phase FIR', 'Location','best');


%% ===== IIR: derive poles, zeros, transfer function =====
N = 3;
[b,a] = butter(N, Wn, 'high');
[z,p,k] = tf2zpk(b,a);              % zeros, poles, gain

fprintf('\n===== IIR Butterworth, order %d =====\n', N);
fprintf('Number of poles: %d,  Number of zeros: %d\n', numel(p), numel(z));
fprintf('Zeros (z-plane):\n');   disp(z);
fprintf('  |zeros| = '); fprintf('%.4f ', abs(z)); fprintf('\n');
fprintf('Poles (z-plane):\n');   disp(p);
fprintf('  |poles| = '); fprintf('%.4f ', abs(p)); fprintf(' (stable if <1)\n');
fprintf('Gain k = %.5f\n', k);

% numerator/denominator coefficients (the transfer function H(z))
fprintf('b (numerator)  : '); fprintf('%.5f ', b); fprintf('\n');
fprintf('a (denominator): '); fprintf('%.5f ', a); fprintf('\n');

% pretty-print H(z) symbolically if Symbolic Toolbox is available
if exist('sym','file')
    syms z_
    num = poly2sym(b, z_);         % builds b(1)*z^n + ...  (positive powers)
    den = poly2sym(a, z_);
    Hz  = num/den;
    fprintf('H(z) = \n'); pretty(vpa(Hz, 5));
end

%% ===== Settling time (samples) — all filters except the 11-tap FIR =====
% Settle = last sample whose |h| exceeds thresh*peak.
% Causal filters (IIR, min-phase): one-sided settle, measured from their start.
% Zero-phase (filtfilt): symmetric, so report pre-impulse (lookahead), post, and total.

fprintf('\n===== Settling time (thresh = %.0e of peak, fs = %d) =====\n', thresh, fs);

% --- helper: one-sided settle for a causal response that starts at index i0 ---
one_sided_settle = @(h, i0) ...
    ( find(abs(h) > thresh*max(abs(h)), 1, 'last') - i0 + 1 );

% --- 2nd-order causal IIR (h_causal): impulse is at index m, response starts there ---
%s2 = one_sided_settle(h_causal, m);
%fprintf('2nd-order IIR (causal)   : %4d samples = %.3f ms (one-sided)\n', ...
%    s2, s2/fs*1000);

% --- 3rd-order causal IIR (h1_casual) ---
s3 = one_sided_settle(h1_casual, m);
fprintf('3rd-order IIR (causal)   : %4d samples = %.3f ms (one-sided)\n', ...
    s3, s3/fs*1000);

% --- min-phase FIR (h_min): causal, starts at its first tap (index 1) ---
smin = one_sided_settle(h_min, 1);
fprintf('min-phase FIR (causal)   : %4d samples = %.3f ms (one-sided)\n', ...
    smin, smin/fs*1000);

% --- zero-phase filtfilt (h_ff): symmetric about the impulse at index m ---
pk_ff = max(abs(h_ff));
idx   = find(abs(h_ff) > thresh*pk_ff);   % all samples above threshold
pre   = m - idx(1);          % samples BEFORE impulse  -> lookahead
post  = idx(end) - m;        % samples AFTER impulse
tot   = idx(end) - idx(1) + 1;
fprintf('zero-phase (filtfilt)    : total=%4d samples = %.3f ms\n', tot, tot/fs*1000);
fprintf('                           lookahead(pre)=%d, post=%d samples\n', pre, post);
fprintf('   -> buffer length = %d, lookahead = %d\n', tot, pre);