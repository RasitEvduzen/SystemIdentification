clc; clear all; close all;
% =========================================================================
%  Input Signal Comparison — Mass-Spring-Damper  (Stage 1)
%
%  For each excitation type one combined figure is produced (3×3 layout):
%
%    [1] Train input signal    [2] Train output signal    [3] Train fit
%    [4] Validation fit        [5] Learning curve         [6] Test fit
%    [7] Bode magnitude        [8] Bode phase             [9] Coherence
%
%  Bode:
%    Red  solid  — True system (analytical transfer function)
%    Black dashed — NN model   (ETFE via Welch / tfestimate)
%
%  Coherence:  γ²(f) = |P_uy|² / (P_uu · P_yy)
%    Measures how linearly the input explains the output at each frequency.
%    γ² < 0.8 (red dashed) → poor excitation at that frequency.
%
%  Required files (same folder or MATLAB path):
%    SystemDef.m  |  SignalGenerator.m  |  DataSet.m  |  NeuralNet.m
% =========================================================================

%% ── 1.  System Definition ────────────────────────────────────────────────
p = struct('m',1.0, 'c',0.5, 'k',2.0);

dynamics  = @(s,u,p) [ s(2);
                        (u - p.c*s(2) - p.k*s(1)) / p.m ];
outputFcn = @(s,p)    s(1);

sys = SystemDef(dynamics, outputFcn, p, ...
                'nx',2, 'ny',1, 'dt',1e-2, 'Tsim',50, ...
                'name','Mass-Spring-Damper');

%  Analytical transfer function  H(jω) = 1 / (-m·ω² + j·c·ω + k)
%  Passed as handle so plotCombined stays system-agnostic.
%  Set to [] to skip true-system overlay on Bode.
trueTF = @(omega) 1 ./ (-p.m*omega.^2 + 1j*p.c*omega + p.k);

%% ── 2.  Signal definitions ───────────────────────────────────────────────
signals = {
    SignalGenerator('dirac', 'amplitude', 1.0)
    SignalGenerator('step',  'amplitude', 1.0)
    SignalGenerator('prbs',  'amplitude', 2.0, 'prbsPeriod', 0.5)
    SignalGenerator('sweep', 'amplitude', 1.5, 'f0', 0.01, 'f1', 10.0)
};

%% ── 3.  Loop: train + plot each input ────────────────────────────────────
results = struct();

for i = 1:numel(signals)
    sig   = signals{i};
    label = sig.type;

    fprintf('\n%s\n', repmat('━',1,65));
    fprintf('  INPUT: %s\n', upper(label));
    fprintf('%s\n', repmat('━',1,65));

    % ── Dataset ────────────────────────────────────────────────────────
    ds = DataSet(sys, sig, 'ky',2, 'ku',1);
    ds.build('seed',42, 'valSplit',0.5);

    % ── Model ──────────────────────────────────────────────────────────
    model = NeuralNet(10, 'tanh');
    model.compile('optimizer','lm', 'loss','sse', 'mu0',1, 'epochs',200);
    history = model.fit(ds.Xtr, ds.Ytr, ...
                        'validation_data',{ds.Xva, ds.Yva}, 'seed',1);

    % ── Metrics ────────────────────────────────────────────────────────
    fprintf('\n--- Evaluation [%s] ---\n', label);
    mTR = model.evaluate(ds.Xtr, ds.Ytr, 'Train');
    mVA = model.evaluate(ds.Xva, ds.Yva, 'Validation');
    mTE = model.evaluate(ds.Xte, ds.Yte, 'Test (Chirp)');

    % ── Store ───────────────────────────────────────────────────────────
    results(i).label   = label;
    results(i).rmse_tr = mTR.rmse;
    results(i).rmse_va = mVA.rmse;
    results(i).rmse_te = mTE.rmse;

    % ── Figure ──────────────────────────────────────────────────────────
    plotCombined(model, history, ds, trueTF, label, sys.dt);
end

%% ── 4.  Summary table ────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('═',1,65));
fprintf('  COMPARISON SUMMARY\n');
fprintf('%s\n', repmat('─',1,65));
fprintf('  %-10s  %14s  %14s  %14s\n', 'Input','Train RMSE','Val RMSE','Test RMSE');
fprintf('%s\n', repmat('─',1,65));
for i = 1:numel(results)
    fprintf('  %-10s  %14.6f  %14.6f  %14.6f\n', ...
            results(i).label, results(i).rmse_tr, ...
            results(i).rmse_va, results(i).rmse_te);
end
fprintf('%s\n\n', repmat('═',1,65));


%% =========================================================================
%                         PLOT FUNCTION  (3 × 3)
%% =========================================================================
function plotCombined(model, history, ds, trueTF, label, dt)
% ── Predictions ────────────────────────────────────────────────────────
    rmse    = @(a,b) sqrt(mean((a-b).^2));
    yhatTR  = model.predict(ds.Xtr);
    yhatVA  = model.predict(ds.Xva);
    yhatTE  = model.predict(ds.Xte);
    S       = model.arch(2);

    figure('Name', sprintf('Input: %s', upper(label)), ...
           'Color','w', 'NumberTitle','off', ...
           'units','normalized','outerposition',[0 0 1 1]);

    % ── [1]  Train input signal ──────────────────────────────────────────
    subplot(3,3,1)
    plot(ds.t_train, ds.u_train, 'b', 'LineWidth',1.2); grid on;
    xlabel('Time (s)'); ylabel('u(t) [N]');
    title(sprintf('[%s]  Training Input', upper(label)));

    % ── [2]  Train output signal ─────────────────────────────────────────
    subplot(3,3,2)
    plot(ds.t_train, ds.y_train, 'r', 'LineWidth',1.2); grid on;
    xlabel('Time (s)'); ylabel('x(t) [m]');
    title('Training Output  (position)');

    % ── [3]  Train fit ───────────────────────────────────────────────────
    subplot(3,3,3)
    plot(ds.Ytr,  'r',   'LineWidth',1.8); hold on; grid on;
    plot(yhatTR,  'k--', 'LineWidth',1.4);
    legend('True','NN','Interpreter','latex','Location','best');
    xlabel('Step'); ylabel('$x(t)$ [m]','Interpreter','latex');
    title(sprintf('Train   RMSE = %.2e m', rmse(ds.Ytr, yhatTR)));

    % ── [4]  Validation fit ──────────────────────────────────────────────
    subplot(3,3,4)
    plot(ds.Yva,  'r',   'LineWidth',1.8); hold on; grid on;
    plot(yhatVA,  'k--', 'LineWidth',1.4);
    legend('True','NN','Interpreter','latex','Location','best');
    xlabel('Step'); ylabel('$x(t)$ [m]','Interpreter','latex');
    title(sprintf('Validation   RMSE = %.2e m', rmse(ds.Yva, yhatVA)));

    % ── [5]  Learning curve ──────────────────────────────────────────────
    subplot(3,3,5)
    semilogy(history.loss,     'b',   'LineWidth',2); hold on; grid on;
    semilogy(history.val_loss, 'r--', 'LineWidth',2);
    legend('Train loss','Val loss','Interpreter','latex');
    xlabel('Epoch'); ylabel('SSE');
    title(sprintf('Learning Curve  |  S=%d  |  Best Val=%.2e', ...
                  S, history.val_min));

    % ── [6]  Test fit ────────────────────────────────────────────────────
    subplot(3,3,6)
    plot(ds.Yte,  'r', 'LineWidth',2.5); hold on; grid on;
    plot(yhatTE,  'k', 'LineWidth',1.2);
    legend('True','NN','Interpreter','latex','Location','best');
    xlabel('Step'); ylabel('$x(t)$ [m]','Interpreter','latex');
    title(sprintf('Test — Chirp   RMSE = %.2e m', rmse(ds.Yte, yhatTE)));

    % ── Frequency analysis setup ─────────────────────────────────────────
    Fs      = 1 / dt;
    ky      = ds.ky;
    ku      = ds.ku;
    lag     = max(ky, ku);
    N_full  = size(ds.y_train, 2);

    % Full NARX regressors from the training time series
    [X_full, ~] = buildNARX(ds.y_train, ds.u_train, ky, ku, ds.sys.ny);
    y_hat_full  = model.predict(X_full);          % [M × 1]  NN one-step output

    % Align true output and input with NARX window offset
    y_true_seg = ds.y_train(:, lag+1:end)';       % [M × ny]
    u_seg      = ds.u_train(lag+1:end)';           % [M × 1]

    % Welch parameters
    nfft    = min(2048, floor(length(u_seg) / 4));
    nfft    = 2^nextpow2(nfft);                   % power of 2 for speed
    win     = hanning(nfft);
    noverlap= nfft / 2;

    % ── [7]  Bode — Magnitude ────────────────────────────────────────────
    subplot(3,3,7)

    %  NN ETFE: H_nn(f) = cpsd(u,y_nn) / pwelch(u)
    [H_nn, f_bode] = tfestimate(u_seg, y_hat_full, win, noverlap, nfft, Fs);
    semilogx(f_bode, 20*log10(abs(H_nn)), 'k--', 'LineWidth',1.8);
    hold on; grid on;

    %  True analytical (if provided)
    if ~isempty(trueTF)
        omega_true = 2*pi * logspace(log10(0.05), log10(Fs/2), 500);
        H_true     = trueTF(omega_true);
        semilogx(omega_true/(2*pi), 20*log10(abs(H_true)), 'r-', 'LineWidth',2);
        legend('NN (ETFE)','True (analytical)','Interpreter','latex','Location','best');
    else
        legend('NN (ETFE)','Interpreter','latex','Location','best');
    end

    xlabel('Frequency [Hz]'); ylabel('Magnitude [dB]');
    title('Bode — Magnitude');
    xlim([0.05, Fs/2]);

    % ── [8]  Bode — Phase ────────────────────────────────────────────────
    subplot(3,3,8)

    semilogx(f_bode, angle(H_nn)*180/pi, 'k--', 'LineWidth',1.8);
    hold on; grid on;

    if ~isempty(trueTF)
        semilogx(omega_true/(2*pi), angle(H_true)*180/pi, 'r-', 'LineWidth',2);
        legend('NN (ETFE)','True (analytical)','Interpreter','latex','Location','best');
    else
        legend('NN (ETFE)','Interpreter','latex','Location','best');
    end

    xlabel('Frequency [Hz]'); ylabel('Phase [°]');
    title('Bode — Phase');
    xlim([0.05, Fs/2]);
    ylim([-200, 20]);

    % ── [9]  Coherence ───────────────────────────────────────────────────
    subplot(3,3,9)

    [Cxy, f_coh] = mscohere(u_seg, y_true_seg, win, noverlap, nfft, Fs);
    semilogx(f_coh, Cxy, 'b', 'LineWidth',1.8); hold on; grid on;
    yline(0.8, 'r--', 'LineWidth',1.5, 'Label','0.8 threshold');
    xlabel('Frequency [Hz]'); ylabel('\gamma^2 [-]');
    title('Coherence  \gamma^2(u \rightarrow y)');
    xlim([0.05, Fs/2]);
    ylim([0, 1.05]);

    % ── Super title ──────────────────────────────────────────────────────
    sgtitle(sprintf('MSD — NARX SysID  |  Input: %s  |  ky=%d  ku=%d', ...
                    upper(label), ds.ky, ds.ku), ...
            'FontSize',13, 'Interpreter','none', 'FontWeight','bold');
end


%% =========================================================================
%               NARX REGRESSOR BUILDER  (used for frequency analysis)
%% =========================================================================
function [X, Y] = buildNARX(y, u, ky, ku, ny)
% Rebuild NARX regressors from a full time series
%   y  : [ny × N]  output
%   u  : [1  × N]  input
%   Returns X [M × (ny*ky+ku)],  Y [M × ny]
    lag = max(ky, ku);
    N   = size(y, 2);
    M   = N - lag;
    R   = ny*ky + ku;
    X   = zeros(M, R);
    Y   = zeros(M, ny);
    for i = 1:M
        t   = i + lag;
        col = 1;
        for k = 1:ky
            X(i, col:col+ny-1) = y(:, t-k)';
            col = col + ny;
        end
        for k = 1:ku
            X(i, col) = u(t-k);
            col = col + 1;
        end
        Y(i,:) = y(:,t)';
    end
end