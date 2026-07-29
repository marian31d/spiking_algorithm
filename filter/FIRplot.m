%% FIR options vs causal IIR — design and compare
fs    = 20000;
fc    = 300;
order = 3;

% --- reference: causal IIR Butterworth ---
[b_iir, a_iir] = butter(order, fc/(fs/2), 'high');

% ---------------------------------------------------------------
% 1) LONG SYMMETRIC (linear-phase) FIR — matches Butterworth magnitude
%    but needs many taps -> large constant delay (Ntaps-1)/2
% ---------------------------------------------------------------
Ntaps_long = 85;                       % ~what's needed for a faithful 300 Hz HP
% design a linear-phase FIR high-pass with a window method
b_fir_sym = fir1(Ntaps_long-1, fc/(fs/2), 'high', hamming(Ntaps_long));
% (symmetric by construction -> linear phase, delay = (Ntaps_long-1)/2)

% ---------------------------------------------------------------
% 2) MINIMUM-PHASE FIR ("non-symmetric") — same magnitude, energy
%    front-loaded -> much smaller group delay, but nonlinear phase
% ---------------------------------------------------------------
% take the linear-phase design and convert to minimum phase.
% firminphase needs the polynomial; it preserves |H(f)| but moves
% the phase to be minimum (lowest possible delay for that magnitude).
b_fir_min = firminphase(b_fir_sym);    % requires DSP System Toolbox
% (if you don't have firminphase, see the note below for an alternative)

% ---------------------------------------------------------------
% 3) SHORT FIR — few taps for ~5-sample delay if symmetric.
%    11 taps -> (11-1)/2 = 5 sample delay, but cutoff is gentle/soft.
% ---------------------------------------------------------------
Ntaps_short = 11;
b_fir_short = fir1(Ntaps_short-1, fc/(fs/2), 'high', hamming(Ntaps_short));

%% ---- impulse responses ----
figure;
stem(0:Ntaps_long-1,  b_fir_sym,  'filled'); hold on;
stem(0:numel(b_fir_min)-1, b_fir_min, 'filled');
stem(0:Ntaps_short-1, b_fir_short,'filled');
grid on; xlim([0 Ntaps_long]);
title('FIR impulse responses (the taps)');
xlabel('tap'); ylabel('coefficient');
legend('symmetric (linear-phase, long)', ...
       'minimum-phase (non-symmetric)', ...
       'short FIR', 'Location','best');
% NOTE how symmetric is centered (energy in the middle -> big delay),
% minimum-phase is front-loaded (energy at start -> small delay).

%% ---- magnitude responses (all four) ----
nfft = 4096;
[Hi, f]  = freqz(b_iir, a_iir, nfft, fs);
[Hs, ~]  = freqz(b_fir_sym,   1, nfft, fs);
[Hm, ~]  = freqz(b_fir_min,   1, nfft, fs);
[Hsh, ~] = freqz(b_fir_short, 1, nfft, fs);

figure;
semilogx(f, 20*log10(abs(Hi)),  'LineWidth',1.5); hold on;
semilogx(f, 20*log10(abs(Hs)),  'LineWidth',1.5);
semilogx(f, 20*log10(abs(Hm)),  'LineWidth',1.5,'LineStyle','--');
semilogx(f, 20*log10(abs(Hsh)), 'LineWidth',1.5);
xline(fc,':k'); yline(-3,':k');
xlim([10 fs/2]); ylim([-60 5]); grid on;
title('Magnitude response');
xlabel('Hz'); ylabel('|H| (dB)');
legend('causal IIR','FIR symmetric','FIR min-phase','FIR short', ...
       'Location','best');
% min-phase magnitude should overlap symmetric exactly (same |H|);
% short FIR has a much gentler/softer cutoff.

%% ---- phase responses ----
figure;
semilogx(f, unwrap(angle(Hi))*180/pi,  'LineWidth',1.5); hold on;
semilogx(f, unwrap(angle(Hs))*180/pi,  'LineWidth',1.5);
semilogx(f, unwrap(angle(Hm))*180/pi,  'LineWidth',1.5,'LineStyle','--');
semilogx(f, unwrap(angle(Hsh))*180/pi, 'LineWidth',1.5);
xline(fc,':k'); xlim([10 fs/2]); grid on;
title('Phase response');
xlabel('Hz'); ylabel('phase (deg)');
legend('causal IIR','FIR symmetric (linear)','FIR min-phase','FIR short', ...
       'Location','best');
% symmetric FIR = straight line (linear phase); min-phase = curved.

%% ---- group delay (the key trade-off) ----
[gi, fg]  = grpdelay(b_iir, a_iir, nfft, fs);
[gs, ~]   = grpdelay(b_fir_sym,   1, nfft, fs);
[gm, ~]   = grpdelay(b_fir_min,   1, nfft, fs);
[gsh, ~]  = grpdelay(b_fir_short, 1, nfft, fs);

figure;
semilogx(fg, gi,  'LineWidth',1.5); hold on;
semilogx(fg, gs,  'LineWidth',1.5);
semilogx(fg, gm,  'LineWidth',1.5,'LineStyle','--');
semilogx(fg, gsh, 'LineWidth',1.5);
xline(fc,':k'); xlim([10 fs/2]); ylim([0 60]); grid on;
title('Group delay — the latency trade-off');
xlabel('Hz'); ylabel('delay (samples)');
legend('causal IIR','FIR symmetric (~42 flat)','FIR min-phase (low)', ...
       'FIR short (~5 flat)','Location','best');

%% report the delays
fprintf('symmetric FIR (%d taps): constant delay = %.1f samples\n', ...
    Ntaps_long, (Ntaps_long-1)/2);
fprintf('short FIR (%d taps): constant delay = %.1f samples\n', ...
    Ntaps_short, (Ntaps_short-1)/2);
fprintf('min-phase FIR: group delay at 1kHz = %.1f samples (much less)\n', ...
    interp1(fg, gm, 1000));