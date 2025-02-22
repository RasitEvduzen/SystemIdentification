clc; clear all; close all;
% CSTR System Simulation for system identification
% Written By: Rasit
% Date: 22-Feb-2025
%% Modeling a Nonlinear System in State Space using RK
% The nonlinear state-space model of the system will be integrated using RK (Runge-Kutta),
% simulated, and the collected data will be used to create a regression model.

umax = 5;       % Amplitude limits
umin = 0;
Tmin = 0;       % Limits for the duration of the input signal (U)
Tmax = 10;

%% RK-based Nonlinear System State-Space Model Simulation
x1 = 0.2;       % Initial conditions of the system states
x2 = 0.1;
x3 = 0.4;
y = x3;         % Defining x3 as the output state
u = 1;          % Applied input signal
Ts = 1e-2;      % Sampling period

% Initialize storage variables
X1(1) = x1;
X2(1) = x2;
X3(1) = x3;
Y(1)  = y;
U(1)  = u;

loop = 1;
k = 1;

while loop
    u = umin + (umax - umin) * rand;  % Randomly generate input signal amplitude
    duration = round(Tmin + (Tmax - Tmin) * rand);  % Random duration for the input signal
    
    for i = 1:duration
        k = k + 1;
        [x1, x2, x3, y] = CSTR(x1, x2, x3, u, Ts);  % Call system dynamics function
        X1(k) = x1;
        X2(k) = x2;
        X3(k) = x3;
        Y(k)  = y;
        U(k)  = u;
    end
    
    if k > 5000  % Stop condition for the simulation
        loop = 0;
    end
end

%% Creating NARX Data Model
% The system is excited by applying a random amplitude and duration input signal
% to cover all modes of operation.

ny = 3;  % Number of output delays
nu = 4;  % Number of input delays
L  = max([nu, ny]);

INPUT  = [];
OUTPUT = [];

for i = 1:length(Y) - L - 1
    INPUT  = [INPUT; [Y(L + i - ny:L + i - 1), U(L + i - nu:L + i - 1)]];
    OUTPUT = [OUTPUT; Y(L + i)];
end

save CSTR.mat INPUT OUTPUT  % Save data for further use

%% PLOT DATA
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')

subplot(5,1,1);
plot(Y);
title('Model Output');

subplot(5,1,2);
plot(U);
title('Input Signal');

subplot(5,1,3);
plot(X1);
title('X1: State');

subplot(5,1,4);
plot(X2);
title('X2: State');

subplot(5,1,5);
plot(X3);
title('X3: State');
