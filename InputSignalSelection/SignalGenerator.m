classdef SignalGenerator < handle
% =========================================================================
%  SignalGenerator  —  Excitation signal generator for system identification
%
%  Supported types:
%    'prbs'   : Pseudo-Random Binary Sequence (broadband, default for SysID)
%    'step'   : Unit step (scaled by amplitude)
%    'dirac'  : Single impulse at t=0
%    'sweep'  : Sine sweep / chirp  (f0 → f1 over Tsim)
%    'sine'   : Fixed-frequency sine wave
%    'custom' : User-supplied function handle  @(t) → u(t)
%
%  USAGE:
%    sig = SignalGenerator('prbs',  'amplitude', 2.0)
%    sig = SignalGenerator('sweep', 'f0',0.01, 'f1',10.0, 'amplitude',1.5)
%    sig = SignalGenerator('custom','customFcn', @(t) sin(2*pi*t))
%
%    u = sig.generate(sys.t, 'seed', 42)   % returns [1 × N] row vector
%
% =========================================================================

    properties (SetAccess = private)
        type        % signal type string
        amplitude   % scaling factor
        f0          % sweep: start frequency [Hz]
        f1          % sweep: end   frequency [Hz]
        freq        % sine:  frequency [Hz]
        prbsPeriod  % prbs:  switching period [s]  (default 0.5 s)
        customFcn   % custom: function handle @(t)→u
    end

    methods (Access = public)

        % ── Constructor ───────────────────────────────────────────────────
        function obj = SignalGenerator(type, varargin)
        % sig = SignalGenerator(type, Name,Value, ...)
            validTypes = {'prbs','step','dirac','sweep','sine','custom'};
            if ~ismember(type, validTypes)
                error('[SignalGenerator] Unknown type: ''%s''. Choose: %s', ...
                      type, strjoin(validTypes,' | '));
            end

            p = inputParser;
            addParameter(p, 'amplitude',   1.0);
            addParameter(p, 'f0',          0.01);     % sweep start [Hz]
            addParameter(p, 'f1',          10.0);     % sweep end   [Hz]
            addParameter(p, 'freq',        1.0);      % sine [Hz]
            addParameter(p, 'prbsPeriod',  0.5);      % [s]
            addParameter(p, 'customFcn',   []);
            parse(p, varargin{:});

            obj.type       = type;
            obj.amplitude  = p.Results.amplitude;
            obj.f0         = p.Results.f0;
            obj.f1         = p.Results.f1;
            obj.freq       = p.Results.freq;
            obj.prbsPeriod = p.Results.prbsPeriod;
            obj.customFcn  = p.Results.customFcn;

            if strcmp(type,'custom') && isempty(obj.customFcn)
                error('[SignalGenerator] ''custom'' type requires ''customFcn'' handle.');
            end

            fprintf('[SignalGenerator]  Type: %-8s  Amplitude: %.2f\n', ...
                    type, obj.amplitude);
        end

        % ── generate ─────────────────────────────────────────────────────
        function u = generate(obj, t, varargin)
        % u = sig.generate(t)
        % u = sig.generate(t, 'seed', 42)
        %
        %   t : time vector [1 × N]  (from sys.t)
        %   u : signal      [1 × N]
            p = inputParser;
            addParameter(p, 'seed', 42);
            parse(p, varargin{:});
            rng(p.Results.seed);

            N    = length(t);
            dt   = t(2) - t(1);
            Tsim = t(end);
            A    = obj.amplitude;

            switch obj.type

                case 'prbs'
                    period = round(obj.prbsPeriod / dt);
                    u      = zeros(1, N);
                    for i  = 1:N
                        if mod(i, period) == 0
                            u(i) = A * 2*(rand - 0.5);   % uniform in [-A, A]
                        else
                            u(i) = u(max(i-1,1));
                        end
                    end

                case 'step'
                    u = A * ones(1, N);

                case 'dirac'
                    u    = zeros(1, N);
                    u(1) = A;

                case 'sweep'
                    u = A * chirp(t, obj.f0, Tsim, obj.f1);

                case 'sine'
                    u = A * sin(2*pi*obj.freq*t);

                case 'custom'
                    u = obj.customFcn(t);
                    u = u(:)';   % ensure row vector
            end
        end

        % ── summary ──────────────────────────────────────────────────────
        function summary(obj)
            fprintf('\n  SignalGenerator\n');
            fprintf('  Type      : %s\n', obj.type);
            fprintf('  Amplitude : %.3f\n', obj.amplitude);
            switch obj.type
                case 'sweep'
                    fprintf('  f0 → f1   : %.3f → %.3f Hz\n', obj.f0, obj.f1);
                case 'sine'
                    fprintf('  Frequency : %.3f Hz\n', obj.freq);
                case 'prbs'
                    fprintf('  Period    : %.3f s\n', obj.prbsPeriod);
            end
        end

    end   % public methods

end   % classdef
