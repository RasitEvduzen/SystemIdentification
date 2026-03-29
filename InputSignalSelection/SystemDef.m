classdef SystemDef < handle
% =========================================================================
%  SystemDef  —  Generalized continuous-time dynamic system definition
%
%  Defines a system in state-space form:
%      sdot = f(s, u, p)     (dynamics,   function handle)
%      y    = h(s, p)        (output map, function handle)
%
%  USAGE — Mass-Spring-Damper example:
%    dynamics  = @(s,u,p) [s(2);  (u - p.c*s(2) - p.k*s(1))/p.m];
%    outputFcn = @(s,p)    s(1);          % measure position only
%    params    = struct('m',1.0,'c',0.5,'k',2.0);
%
%    sys = SystemDef(dynamics, outputFcn, params, ...
%                   'nx',2, 'ny',1, 'dt',0.01, 'Tsim',50);
%
%  Then simulate:
%    [t, y, s] = sys.simulate(u_vector);
%
% =========================================================================

    properties (SetAccess = private)
        dynamics    % f(s, u, p)  →  sdot   [nx × 1]
        outputFcn   % h(s, p)     →  y      [ny × 1]
        params      % struct — physical parameters (user-defined)
        nx          % state  dimension
        ny          % output dimension
        dt          % sample time  [s]
        Tsim        % simulation duration [s]
        t           % time vector  [1 × N]
        N           % number of time steps
        name        % optional system name (for plots/logs)
    end

    methods (Access = public)

        % ── Constructor ───────────────────────────────────────────────────
        function obj = SystemDef(dynamics, outputFcn, params, varargin)
        % sys = SystemDef(dynamics, outputFcn, params,
        %                 'nx',2, 'ny',1, 'dt',0.01, 'Tsim',50, 'name','MSD')
        %
        %   dynamics  : @(s, u, p) → column vector [nx × 1]
        %   outputFcn : @(s, p)    → column vector [ny × 1]  (or scalar)
        %   params    : struct with physical parameters
            p = inputParser;
            addParameter(p, 'nx',   2);
            addParameter(p, 'ny',   1);
            addParameter(p, 'dt',   1e-2);
            addParameter(p, 'Tsim', 50);
            addParameter(p, 'name', 'System');
            parse(p, varargin{:});

            obj.dynamics  = dynamics;
            obj.outputFcn = outputFcn;
            obj.params    = params;
            obj.nx        = p.Results.nx;
            obj.ny        = p.Results.ny;
            obj.dt        = p.Results.dt;
            obj.Tsim      = p.Results.Tsim;
            obj.name      = p.Results.name;
            obj.t         = 0 : obj.dt : obj.Tsim;
            obj.N         = length(obj.t);

            fprintf('[SystemDef]  %s  |  nx=%d  ny=%d  dt=%.3f  Tsim=%.1f s\n', ...
                    obj.name, obj.nx, obj.ny, obj.dt, obj.Tsim);
        end

        % ── simulate ─────────────────────────────────────────────────────
        function [t_out, y_out, s_out] = simulate(obj, u, varargin)
        % [t, y, s] = sys.simulate(u)
        % [t, y, s] = sys.simulate(u, 's0', [0;0])
        %
        %   u     : input vector [1 × N]
        %   s0    : initial state [nx × 1]  (default: zeros)
        %
        %   t_out : time vector           [1 × N]
        %   y_out : output matrix         [ny × N]
        %   s_out : full state matrix     [nx × N]
            p = inputParser;
            addParameter(p, 's0', zeros(obj.nx,1));
            parse(p, varargin{:});
            s0 = p.Results.s0;

            N_u = length(u);
            if N_u ~= obj.N
                error('[SystemDef.simulate] Input length %d ≠ time vector length %d', ...
                      N_u, obj.N);
            end

            s_out = zeros(obj.nx, obj.N);
            y_out = zeros(obj.ny, obj.N);
            s_out(:,1) = s0;
            y_out(:,1) = obj.evalOutput(s0);

            for i = 1:obj.N-1
                s_out(:,i+1) = obj.rk4step(s_out(:,i), u(i));
                y_out(:,i+1) = obj.evalOutput(s_out(:,i+1));
            end

            t_out = obj.t;
        end

        % ── summary ──────────────────────────────────────────────────────
        function summary(obj)
            fprintf('\n%s\n', repmat('═',1,50));
            fprintf('  SystemDef : %s\n', obj.name);
            fprintf('%s\n', repmat('─',1,50));
            fprintf('  States (nx)  : %d\n', obj.nx);
            fprintf('  Outputs (ny) : %d\n', obj.ny);
            fprintf('  dt           : %.4f s\n', obj.dt);
            fprintf('  Tsim         : %.1f s\n', obj.Tsim);
            fprintf('  N steps      : %d\n', obj.N);
            fprintf('%s\n\n', repmat('═',1,50));
        end

    end   % public methods


    methods (Access = private)

        % ── rk4step ──────────────────────────────────────────────────────
        function sn = rk4step(obj, s, u_i)
        % One RK4 integration step:  s(t+dt) = rk4(s(t), u(t))
            f  = obj.dynamics;
            pr = obj.params;
            h  = obj.dt;
            k1 = f(s,           u_i, pr);
            k2 = f(s + h/2*k1,  u_i, pr);
            k3 = f(s + h/2*k2,  u_i, pr);
            k4 = f(s + h*k3,    u_i, pr);
            sn = s + h/6*(k1 + 2*k2 + 2*k3 + k4);
        end

        % ── evalOutput ───────────────────────────────────────────────────
        function y = evalOutput(obj, s)
        % Evaluate output function h(s,p), ensure column vector [ny×1]
            y = obj.outputFcn(s, obj.params);
            y = y(:);
        end

    end   % private methods

end   % classdef
