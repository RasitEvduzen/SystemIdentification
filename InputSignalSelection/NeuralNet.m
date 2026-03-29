classdef NeuralNet < handle
% =========================================================================
%
%  USAGE:
%    model = NeuralNet(10, 'tanh')               % hidden=10, activation
%    model.compile('optimizer','lm','epochs',100) % attach optimizer
%    history = model.fit(Xtr, Ytr, ...)           % train
%    yhat    = model.predict(X)                   % forward pass
%    metrics = model.evaluate(X, Y, 'Test')       % RMSE / MAE
%
% =========================================================================

    % ── Public readable properties ────────────────────────────────────────
    properties (SetAccess = private)
        hiddenSize   (1,1) double                  % S — number of hidden neurons
        activation   (1,:) char                    % 'tanh' (extensible)
        arch         (1,3) double = [0, 0, 1]      % [R, S, 1]  filled on fit()
        weights      struct                         % Wg, bh, Wc, bc
        config       struct                         % optimizer, loss, mu0, epochs
        isFitted     (1,1) logical = false
    end

    % ── Constructor ───────────────────────────────────────────────────────
    methods (Access = public)

        function obj = NeuralNet(hiddenSize, activation)
        % model = NeuralNet(hiddenSize, activation)
        %   hiddenSize : number of hidden neurons  (e.g. 10)
        %   activation : 'tanh'
            if nargin < 2; activation = 'tanh'; end
            obj.hiddenSize = hiddenSize;
            obj.activation = activation;
            obj.config     = struct();
            obj.weights    = struct();
            fprintf('[NeuralNet]  Hidden: %d  |  Activation: %s\n', ...
                    hiddenSize, activation);
        end

        % ── compile ───────────────────────────────────────────────────────
        function compile(obj, varargin)
        % model.compile('optimizer','lm', 'loss','sse', 'mu0',1, 'epochs',100)
        %
        % Supported optimizers : 'lm'  (Levenberg-Marquardt)
        % Supported losses     : 'sse' (sum-squared error — LM canonical)
            p = inputParser;
            addParameter(p, 'optimizer', 'lm');
            addParameter(p, 'loss',      'sse');
            addParameter(p, 'mu0',        1);
            addParameter(p, 'epochs',     100);
            parse(p, varargin{:});
            obj.config = p.Results;
            fprintf('[compile]    Optimizer: %s  |  Loss: %s  |  mu0: %.0e  |  epochs: %d\n', ...
                    obj.config.optimizer, obj.config.loss, ...
                    obj.config.mu0,       obj.config.epochs);
        end

        % ── fit ───────────────────────────────────────────────────────────
        function history = fit(obj, Xtr, Ytr, varargin)
        % history = model.fit(Xtr, Ytr, 'validation_data',{Xva,Yva}, 'seed',1)
        %
        % - Input / output dims inferred from Xtr, Ytr automatically.
        % - Weights re-initialised each call (set seed for reproducibility).
        % - Returns history struct: .loss, .val_loss, .val_min
            p = inputParser;
            addParameter(p, 'validation_data', {});
            addParameter(p, 'seed', 1);
            parse(p, varargin{:});

            % ── Infer architecture from data ──
            R = size(Xtr, 2);   % input dimension
            S = obj.hiddenSize;
            obj.arch = [R, S, 1];

            % ── Initialise weights ──
            rng(p.Results.seed);
            obj.weights.Wg = rand(S,R) - 0.5;   % [S x R]
            obj.weights.bh = rand(S,1) - 0.5;   % [S x 1]
            obj.weights.Wc = rand(1,S) - 0.5;   % [1 x S]
            obj.weights.bc = rand(1,1) - 0.5;   % scalar

            nParams = S*(R+2) + 1;
            fprintf('\n[fit]  Architecture : %d → %d (%s) → 1  |  params: %d\n', ...
                    R, S, obj.activation, nParams);

            % ── Validation data ──
            valData = p.Results.validation_data;
            hasVal  = ~isempty(valData);
            Xva = []; Yva = [];
            if hasVal; Xva = valData{1}; Yva = valData{2}; end

            fprintf('[fit]  Train: %d  |  Val: %d  samples\n', ...
                    size(Xtr,1), size(Xva,1));
            fprintf('%s\n', repmat('─',1,65));

            % ── Dispatch optimizer ──
            switch lower(obj.config.optimizer)
                case 'lm'
                    history = obj.trainLM(Xtr, Ytr, Xva, Yva, hasVal);
                otherwise
                    error('[fit] Unknown optimizer: ''%s''', obj.config.optimizer);
            end

            obj.isFitted = true;
            fprintf('%s\n', repmat('─',1,65));
            fprintf('[fit]  Done.  Best val SSE = %.6f\n', history.val_min);
        end

        % ── predict ───────────────────────────────────────────────────────
        function yhat = predict(obj, X)
        % yhat = model.predict(X)
        %   X    [N x R]  —  input matrix
        %   yhat [N x 1]  —  model output
            obj.checkFitted();
            yhat = obj.forwardPass(obj.weights, X);
        end

        % ── evaluate ──────────────────────────────────────────────────────
        function metrics = evaluate(obj, X, Y, label)
        % metrics = model.evaluate(X, Y, 'Test')
        %   Prints and returns RMSE, MAE, SSE for the given split.
            obj.checkFitted();
            if nargin < 4; label = 'Set'; end
            yhat         = obj.predict(X);
            metrics.rmse = sqrt(mean((Y - yhat).^2));
            metrics.mae  = mean(abs(Y - yhat));
            metrics.sse  = sum((Y - yhat).^2);
            fprintf('  %-24s  RMSE = %.6f m   MAE = %.6f m\n', ...
                    [label ':'], metrics.rmse, metrics.mae);
        end

        % ── summary ───────────────────────────────────────────────────────
        function summary(obj)
        % model.summary()  — prints architecture table (like Keras)
            R = obj.arch(1); S = obj.arch(2);
            fprintf('\n%s\n', repmat('═',1,50));
            fprintf('  NeuralNet Summary\n');
            fprintf('%s\n', repmat('─',1,50));
            fprintf('  %-18s  %-12s  %s\n','Layer','Output Shape','Params');
            fprintf('%s\n', repmat('─',1,50));
            fprintf('  %-18s  %-12s  %d\n', ...
                    'Input',          sprintf('(N, %d)', R),  0);
            fprintf('  %-18s  %-12s  %d\n', ...
                    sprintf('Dense [%s]', obj.activation), ...
                    sprintf('(N, %d)', S),  S*(R+1));
            fprintf('  %-18s  %-12s  %d\n', ...
                    'Dense [linear]', '(N, 1)',  S+1);
            fprintf('%s\n', repmat('─',1,50));
            fprintf('  Total params : %d\n', S*(R+2)+1);
            fprintf('  Optimizer    : %s\n', obj.config.optimizer);
            fprintf('  Loss         : %s\n', obj.config.loss);
            fprintf('%s\n\n', repmat('═',1,50));
        end

    end   % public methods


    % ── Private / engine methods ──────────────────────────────────────────
    methods (Access = private)

        % ── checkFitted ───────────────────────────────────────────────────
        function checkFitted(obj)
            if ~obj.isFitted
                error('[NeuralNet] Model not trained yet. Call model.fit() first.');
            end
        end

        % ── forwardPass ───────────────────────────────────────────────────
        function yhat = forwardPass(~, w, X)
        % Raw forward pass on a weight struct  (used inside trainLM)
        %   A    = tanh(X * Wg' + bh')   [N x S]
        %   yhat = A  * Wc'  + bc        [N x 1]
            A    = tanh(X * w.Wg' + w.bh');
            yhat = A  * w.Wc'    + w.bc;
        end

        % ── trainLM ───────────────────────────────────────────────────────
        function history = trainLM(obj, Xtr, Ytr, Xva, Yva, hasVal)
        % Levenberg-Marquardt engine
        %   Δθ = -(JᵀJ + μI)⁻¹ Jᵀe
        %   μ ×0.1  on accepted step  → Gauss-Newton direction
        %   μ ×10   on rejected step  → Gradient Descent direction
            w      = obj.weights;
            S      = obj.arch(2);
            R      = obj.arch(1);
            Nmax   = obj.config.epochs;
            mu     = obj.config.mu0;
            I_mat  = eye(S*(R+2)+1);

            FvalMIN = inf;
            xbest   = obj.pack(w);
            Ftr     = zeros(1, Nmax);
            Fva     = zeros(1, Nmax);
            iter    = 0;
            running = true;

            while running
                iter = iter + 1;

                % ── Residual ──
                yhat = obj.forwardPass(w, Xtr);
                e    = Ytr - yhat;
                f    = e'  * e;

                % ── Jacobian ──
                J = obj.jacobian(Xtr, w, S, R);

                % ── Inner damping loop ──
                inner = true;
                while inner
                    p  = -(J'*J + mu*I_mat) \ (J'*e);
                    xv = obj.pack(w);
                    wz = obj.unpack(xv + p, S, R);
                    fz = norm(Ytr - obj.forwardPass(wz, Xtr))^2;

                    if fz < f
                        w     = obj.unpack(xv + p, S, R);
                        mu    = max(mu * 0.1, 1e-20);
                        inner = false;
                    else
                        mu = mu * 10;
                        if mu > 1e20
                            fprintf('  [LM] mu > 1e20 — no descent direction, stopping at epoch %d.\n', iter);
                            inner = false; running = false;
                        end
                    end
                end

                % ── Record losses ──
                yhat      = obj.forwardPass(w, Xtr);
                e         = Ytr - yhat;
                Ftr(iter) = e' * e;
                if hasVal
                    Fva(iter) = norm(Yva - obj.forwardPass(w, Xva))^2;
                end

                % ── Best-val checkpoint ──
                if hasVal && Fva(iter) < FvalMIN
                    xbest   = obj.pack(w);
                    FvalMIN = Fva(iter);
                end

                fprintf('Epoch %4d/%d   ||g|| %.2e   loss %.4e   val_loss %.4e\n', ...
                        iter, Nmax, norm(2*J'*e), Ftr(iter), Fva(iter));

                if iter >= Nmax; running = false; end
            end

            % ── Restore best weights ──
            if hasVal
                obj.weights = obj.unpack(xbest, S, R);
            else
                obj.weights = w;
            end

            history.loss     = Ftr(1:iter);
            history.val_loss = Fva(1:iter);
            history.val_min  = FvalMIN;
        end

        % ── jacobian ─────────────────────────────────────────────────────
        function J = jacobian(~, T, w, S, R)
        % Analytical Jacobian — fully vectorised over N samples
        % Layout: [vec(Wg) | bh | Wc | bc]   (column-major Wg)
        %
        %   d(-e_i)/d(bc)        = -1
        %   d(-e_i)/d(Wc_j)      = -a_j
        %   d(-e_i)/d(bh_j)      = -Wc_j * (1 - a_j²)
        %   d(-e_i)/d(Wg_{j,r})  = -Wc_j * T_{i,r} * (1 - a_j²)
            N_params = S*(R+2) + 1;
            N        = size(T, 1);
            J        = zeros(N, N_params);

            idx_bc = N_params;
            idx_Wc = S*(R+1)+1 : S*(R+2);
            idx_bh = S*R+1     : S*R+S;

            A    = tanh(T * w.Wg' + w.bh');   % [N x S]
            dA   = 1 - A.^2;                  % [N x S]

            J(:, idx_bc) = -1;
            J(:, idx_Wc) = -A;
            J(:, idx_bh) = -(dA .* w.Wc);

            WcDa = dA .* w.Wc;
            for r = 1:R
                J(:, (r-1)*S+1:r*S) = -WcDa .* T(:,r);
            end
        end

        % ── pack / unpack ────────────────────────────────────────────────
        function x = pack(~, w)
        % Flatten weight struct → column vector  (column-major Wg)
            x = [w.Wg(:); w.bh; w.Wc'; w.bc];
        end

        function w = unpack(~, x, S, R)
        % Rebuild weight struct from column vector
            w.Wg = reshape(x(1:S*R),             S, R);
            w.bh =         x(S*R+1    : S*R+S);
            w.Wc =         x(S*(R+1)+1: S*(R+2))';
            w.bc =         x(S*(R+2)+1);
        end

    end   % private methods

end   % classdef
