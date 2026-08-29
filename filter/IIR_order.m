% Zero-phase (filtfilt) vs causal Butterworth high-pass: impulse response
fs     = 20000;      % sampling rate (Hz)
fc     = 300;        % high-pass cutoff (Hz)
order  = 2;          % Butterworth order
order1 = 3
thresh = 1e-3;       % settle threshold (fraction of peak)

Wn = fc/(fs/2);                        % cutoff normalized to Nyquist (MATLAB convention)
[b,a] = butter(order, Wn, 'high');     % 3rd-order Butterworth high-pass
[d,c] = butter(order1, Wn, 'high');     % 3rd-order Butterworth high-pass
NT = 2001;                             % buffer length (odd -> has a true center)
m  = (NT+1)/2;                         % center index
x  = zeros(NT,1);  x(m) = 1;           % unit impulse at the CENTER

h_causal = filter(b, a, x);            % one-directional (causal) response
h1_casual = filter(d, c, x)
t = (1:NT)' - m;                       % samples relative to impulse (0 = impulse)

% --- frequency response ---
nfft = 4096;
[H, f]   = freqz(b, a, nfft, fs);          % single-pass complex response
mag_db  = 20*log10(abs(H));               % filter (causal) magnitude
phase   = unwrap(angle(H))*180/pi;        % filter phase
[gd, fg] = grpdelay(b, a, nfft, fs);       % filter group delay (samples)

nfft = 4096;
[H1, f1]   = freqz(d, c, nfft, fs);          % single-pass complex response
mag1_db  = 20*log10(abs(H1));               % filter (causal) magnitude
phase1   = unwrap(angle(H1))*180/pi;        % filter phase
[gd1, fg1] = grpdelay(d, c, nfft, fs);       % filter group delay (samples)

%% Figure 1: Impulse Response
figure;
plot(t, h_causal, 'LineWidth', 1.2); hold on;
plot(t, h1_casual, 'LineWidth', 1.2);
xline(0,'k:');
legend('IIR Butterworth 2nd order','IIR Butterworth 3rd order');
xlabel('samples relative to impulse'); ylabel('amplitude');
title('Impulse response: 2nd vs 3rd order Butterworth'); grid on;

%% Figure 2: Bode magnitude (both)
figure;
semilogx(f, mag_db,'LineWidth',1.5); hold on;
semilogx(f, mag1_db,'LineWidth',1.5);
xline(fc,':k'); yline(-3,':k');
xlim([10 fs/2]); grid on;
title('Bode magnitude');
xlabel('Hz'); ylabel('|H| (dB)');
legend('2nd order','3rd order','Location','best');

%% Figure 3: Bode phase (both)
figure;
semilogx(f, phase,'LineWidth',1.5); hold on;
semilogx(f, phase1,'LineWidth',1.5);
xline(fc,':k');
xlim([10 fs/2]); grid on;
title('Bode phase');
xlabel('Hz'); ylabel('phase (deg)');
legend('2nd order','3rd order','Location','best');

%% --- group delay vs frequency (the phase nonlinearity) ---
figure;
plot(fg, gd, 'LineWidth',1.5); hold on;
plot(fg1, gd1, 'LineWidth', 1.5);
xlim([10 3000]);
ylim([0 35]);  
grid on;
title('Group delay');
xlabel('Hz'); ylabel('delay (samples)');
legend('2nd order','3rd order','Location','best');