%% Filter analysis: causal IIR (filter) vs zero-phase (filtfilt) — separate figures
fs    = 20000;          % sampling rate (Hz)
fc    = 300;            % high-pass cutoff (Hz)
order = 3;              % filter order

% --- design the IIR Butterworth high-pass ---
Wn = fc / (fs/2);
[b, a] = butter(order, Wn, 'high');
disp('b ='); disp(b);
disp('a ='); disp(a);

% --- impulse responses ---
N = 200;
imp = zeros(N,1);  imp(N/2 + 1) = 1.0;
y_filter   = filter(b, a, imp);     % causal (your chip, == scipy lfilter)
y_filtfilt = filtfilt(b, a, imp);   % zero-phase (kilosort, == scipy filtfilt)
t = (0:N-1).' - N/2;

% --- template-5 waveform ---
wave = wave(:);
sx = (0:numel(wave)-1).';
sl = filter(b, a, wave);
sf = filtfilt(b, a, wave);

% --- frequency response ---
nfft = 4096;
[H, f]   = freqz(b, a, nfft, fs);          % single-pass complex response
mag1_db  = 20*log10(abs(H));               % filter (causal) magnitude
phase1   = unwrap(angle(H))*180/pi;        % filter phase
[gd, fg] = grpdelay(b, a, nfft, fs);       % filter group delay (samples)

% filtfilt response: magnitude squared (filter applied twice), phase zero
mag2_db   = 2*mag1_db;                      % |H|^2 in dB  == 2*|H|_dB
phase2    = zeros(size(phase1));            % zero phase

%% Figure 1: impulse response
figure;
plot(t, y_filter,'LineWidth',1.5); hold on;
plot(t, y_filtfilt,'LineWidth',1.5);
xline(0,':k'); xlim([-30 50]); grid on;
title('Impulse response');
xlabel('samples rel. to impulse'); ylabel('amplitude');
legend('filter (causal)','filtfilt (zero-phase)','Location','best');

%% Figure 2: template waveform after each filter
figure;
plot(sx, wave,'b--','LineWidth',1); hold on;
plot(sx, sl,'LineWidth',1.5);
plot(sx, sf,'LineWidth',1.5);
xline(20,':k'); xlim([0 2000]); grid on;
title('Channel 45, mP79: shape after each filter');
xlabel('sample'); ylabel('whitened amplitude');
legend('original','filter (causal)','filtfilt','Location','best');

%% Figure 3: Bode magnitude (both)
figure;
semilogx(f, mag1_db,'LineWidth',1.5); hold on;
semilogx(f, mag2_db,'LineWidth',1.5);
xline(fc,':k'); yline(-3,':k');
xlim([10 fs/2]); grid on;
title('Bode magnitude');
xlabel('Hz'); ylabel('|H| (dB)');
legend('filter (causal)','filtfilt (|H|^2)','Location','best');

%% Figure 4: Bode phase (both)
figure;
semilogx(f, phase1,'LineWidth',1.5); hold on;
semilogx(f, phase2,'LineWidth',1.5);
xline(fc,':k');
xlim([10 fs/2]); grid on;
title('Bode phase');
xlabel('Hz'); ylabel('phase (deg)');
legend('filter (causal)','filtfilt (zero phase)','Location','best');

%% --- group delay vs frequency (the phase nonlinearity) ---
figure;
plot(fg, gd, 'LineWidth',1.5);
xlim([10 3000]);
ylim([0 35]);  
grid on;
title('Group delay of the IIR filter (filter)  --  filtfilt would be flat 0');
xlabel('Hz'); ylabel('delay (samples)');

%% report
[~, il]  = max(abs(y_filter));
[~, iff] = max(abs(y_filtfilt));
fprintf('impulse peak: filter @ %d samp, filtfilt @ %d samp\n', t(il), t(iff));
fprintf('group delay (causal): 300Hz=%.1f, 1kHz=%.1f, 2kHz=%.1f samp\n', ...
    interp1(fg,gd,300), interp1(fg,gd,1000), interp1(fg,gd,2000));
fprintf('phase @10Hz = %.1f, @Nyquist = %.1f, excursion = %.1f deg\n', ...
    phase1(1), phase1(end), phase1(1) - phase1(end));