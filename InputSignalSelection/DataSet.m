classdef DataSet < handle
% =========================================================================
%  DataSet  —  System simulation + NARX window builder + data splitter
%
%  Pipeline (called by build()):
%    1. Generate training input  via SignalGenerator
%    2. Simulate system          via SystemDef.simulate()
%    3. Build NARX regressor matrix
%    4. Split → Train / Validation  (alternating-index)
%    5. Generate fixed test signal (chirp) → Test split
%
%  NARX structure  (ky past outputs, ku past inputs → 1 step ahead output)
%    φ(t) = [y(t-1) … y(t-ky)  |  u(t-1) … u(t-ku)]
%    target: y(t)
%    Input dimension R = ny*ky + ku   (fed automatically to NeuralNet)
%
%  USAGE:
%    ds = DataSet(sys, trainSig, 'ky',2, 'ku',2)
%    ds.build('seed',42, 'valSplit',0.5)
%    model.fit(ds.Xtr, ds.Ytr, 'validation_data',{ds.Xva, ds.Yva})
%    model.evaluate(ds.Xte, ds.Yte, 'Test')
%    ds.plotSignals()
%
% =========================================================================

    properties (SetAccess = private)
        % ── Configuration ─────────────────────────────────────────────
        sys         % SystemDef handle
        trainSig    % SignalGenerator handle  (training)
        testSig     % SignalGenerator handle  (test, default: sweep)

        ky          % number of past output lags
        ku          % number of past input  lags

        % ── Simulation results ────────────────────────────────────────
        t_train     % time vector          [1 × N]
        u_train     % training input       [1 × N]
        y_train     % training output      [ny × N]
        s_train     % training states      [nx × N]

        t_test      % time vector          [1 × N]
        u_test      % test input           [1 × N]
        y_test      % test output          [ny × N]
        s_test      % test states          [nx × N]

        % ── Dataset splits ────────────────────────────────────────────
        Xtr         % train  regressors   [Ntr × R]
        Ytr         % train  targets      [Ntr × ny]
        Xva         % val    regressors   [Nva × R]
        Yva         % val    targets      [Nva × ny]
        Xte         % test   regressors   [Nte × R]
        Yte         % test   targets      [Nte × ny]

        inputDim    % R = ny*ky + ku
        isBuilt     (1,1) logical = false
    end

    methods (Access = public)

        % ── Constructor ───────────────────────────────────────────────────
        function obj = DataSet(sys, trainSig, varargin)
        % ds = DataSet(sys, trainSig, 'ky',2, 'ku',2, 'testSig', sigObj)
        %
        %   sys      : SystemDef handle
        %   trainSig : SignalGenerator handle for training
        %   'ky'     : output lags  (default 1)
        %   'ku'     : input  lags  (default 1)
        %   'testSig': SignalGenerator for test (default: chirp sweep)
            p = inputParser;
            addParameter(p, 'ky',      1);
            addParameter(p, 'ku',      1);
            addParameter(p, 'testSig', []);
            parse(p, varargin{:});

            obj.sys      = sys;
            obj.trainSig = trainSig;
            obj.ky       = p.Results.ky;
            obj.ku       = p.Results.ku;

            % Default test signal: chirp sweep
            if isempty(p.Results.testSig)
                obj.testSig = SignalGenerator('sweep', ...
                                              'f0',       0.01, ...
                                              'f1',       10.0, ...
                                              'amplitude', 1.5);
            else
                obj.testSig = p.Results.testSig;
            end

            obj.inputDim = sys.ny * obj.ky + obj.ku;

            fprintf('[DataSet]    ky=%d  ku=%d  →  input dim R=%d\n', ...
                    obj.ky, obj.ku, obj.inputDim);
        end

        % ── build ─────────────────────────────────────────────────────────
        function build(obj, varargin)
        % ds.build('seed',42, 'valSplit',0.5)
        %
        % Runs full pipeline:
        %   simulate train → build NARX → split train/val
        %   simulate test  → build NARX → test set
            p = inputParser;
            addParameter(p, 'seed',     42);
            addParameter(p, 'valSplit', 0.5);
            parse(p, varargin{:});
            seed    = p.Results.seed;
            valFrac = p.Results.valSplit;

            fprintf('\n[DataSet.build]  Simulating training signal...\n');

            % ── 1. Simulate training ──────────────────────────────────────
            u_tr = obj.trainSig.generate(obj.sys.t, 'seed', seed);
            [obj.t_train, obj.y_train, obj.s_train] = obj.sys.simulate(u_tr);
            obj.u_train = u_tr;

            % ── 2. Build NARX windows from training simulation ────────────
            [X_all, Y_all] = obj.buildNARX(obj.y_train, obj.u_train);
            N = size(X_all, 1);

            % ── 3. Train / Val split (alternating index) ──────────────────
            stride = round(1 / valFrac);
            idxVa  = stride : stride : N;
            idxTr  = setdiff(1:N, idxVa);
            obj.Xtr = X_all(idxTr,:);  obj.Ytr = Y_all(idxTr,:);
            obj.Xva = X_all(idxVa,:);  obj.Yva = Y_all(idxVa,:);

            % ── 4. Simulate test ──────────────────────────────────────────
            fprintf('[DataSet.build]  Simulating test signal...\n');
            u_te = obj.testSig.generate(obj.sys.t, 'seed', seed+1);
            [obj.t_test, obj.y_test, obj.s_test] = obj.sys.simulate(u_te);
            obj.u_test = u_te;

            % ── 5. Build NARX windows from test simulation ────────────────
            [obj.Xte, obj.Yte] = obj.buildNARX(obj.y_test, obj.u_test);

            obj.isBuilt = true;

            fprintf('[DataSet.build]  Done.\n');
            obj.summary();
        end

        % ── summary ───────────────────────────────────────────────────────
        function summary(obj)
            obj.checkBuilt();
            fprintf('\n%s\n', repmat('═',1,55));
            fprintf('  DataSet Summary\n');
            fprintf('%s\n', repmat('─',1,55));
            fprintf('  System       : %s\n',  obj.sys.name);
            fprintf('  Train signal : %s\n',  obj.trainSig.type);
            fprintf('  Test signal  : %s\n',  obj.testSig.type);
            fprintf('  ky (output lags) : %d\n', obj.ky);
            fprintf('  ku (input  lags) : %d\n', obj.ku);
            fprintf('  Input dim  R : %d  (ny*ky + ku = %d*%d + %d)\n', ...
                    obj.inputDim, obj.sys.ny, obj.ky, obj.ku);
            fprintf('%s\n', repmat('─',1,55));
            fprintf('  %-12s  %5d samples  [%d × %d]\n', 'Train:',  size(obj.Xtr,1), size(obj.Xtr,1), size(obj.Xtr,2));
            fprintf('  %-12s  %5d samples  [%d × %d]\n', 'Val:',    size(obj.Xva,1), size(obj.Xva,1), size(obj.Xva,2));
            fprintf('  %-12s  %5d samples  [%d × %d]\n', 'Test:',   size(obj.Xte,1), size(obj.Xte,1), size(obj.Xte,2));
            fprintf('%s\n\n', repmat('═',1,55));
        end

        % ── plotSignals ───────────────────────────────────────────────────
        function plotSignals(obj)
        % ds.plotSignals()  —  plot input and output signals for train & test
            obj.checkBuilt();
            figure('Name','DataSet Signals','Color','w', ...
                   'units','normalized','outerposition',[0 0 1 0.6]);

            subplot(2,2,1)
            plot(obj.t_train, obj.u_train, 'b', 'LineWidth',1.2); grid on;
            xlabel('Time (s)'); ylabel('u(t)');
            title(sprintf('Train Input  [%s]', obj.trainSig.type));

            subplot(2,2,2)
            plot(obj.t_train, obj.y_train, 'r', 'LineWidth',1.2); grid on;
            xlabel('Time (s)'); ylabel('y(t)');
            title('Train Output');

            subplot(2,2,3)
            plot(obj.t_test, obj.u_test, 'b', 'LineWidth',1.2); grid on;
            xlabel('Time (s)'); ylabel('u(t)');
            title(sprintf('Test Input  [%s]', obj.testSig.type));

            subplot(2,2,4)
            plot(obj.t_test, obj.y_test, 'r', 'LineWidth',1.2); grid on;
            xlabel('Time (s)'); ylabel('y(t)');
            title('Test Output');

            sgtitle(sprintf('%s — Signals', obj.sys.name), 'FontSize',13);
        end

    end   % public methods


    methods (Access = private)

        % ── buildNARX ────────────────────────────────────────────────────
        function [X, Y] = buildNARX(obj, y, u)
        % Build NARX regressor matrix from output y [ny × N] and input u [1 × N]
        %
        % Regressor at time t  (1-indexed, t = lag+1 … N):
        %   φ(t) = [y(t-1) … y(t-ky)  |  u(t-1) … u(t-ku)]
        %   target: y(t)
        %
        % Output:
        %   X  [M × (ny*ky + ku)]  regressor matrix
        %   Y  [M × ny]            target matrix
        %   where M = N - lag_max

            ny      = obj.sys.ny;
            ky      = obj.ky;
            ku      = obj.ku;
            lag_max = max(ky, ku);
            N       = size(y, 2);          % number of time steps
            M       = N - lag_max;         % valid samples

            R = ny*ky + ku;
            X = zeros(M, R);
            Y = zeros(M, ny);

            for i = 1:M
                t = i + lag_max;           % current time index (1-based)

                % past outputs: y(t-1), y(t-2), ..., y(t-ky)   [ny each]
                col = 1;
                for k = 1:ky
                    X(i, col:col+ny-1) = y(:, t-k)';
                    col = col + ny;
                end

                % past inputs: u(t-1), u(t-2), ..., u(t-ku)
                for k = 1:ku
                    X(i, col) = u(t-k);
                    col = col + 1;
                end

                % target
                Y(i,:) = y(:, t)';
            end
        end

        % ── checkBuilt ───────────────────────────────────────────────────
        function checkBuilt(obj)
            if ~obj.isBuilt
                error('[DataSet] Not built yet. Call ds.build() first.');
            end
        end

    end   % private methods

end   % classdef
