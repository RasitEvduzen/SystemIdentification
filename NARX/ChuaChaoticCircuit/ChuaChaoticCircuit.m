% Chua Chaotic Circuit Nonlinear ARX Model Identification via ELM
% Written By: Rasit
% 01-Aug-2024
clc, clear, close all
%% 
% Model Parameters
alpha = 9;
beta = 14.3;
gamma = 0;
m0 = -1.143;
m1 = -0.714;

tspan = [0, 50];
x0 = [0.7, 0, 0]; % Initial conditions
Ts = 1e-2;
[t, X] = runge_kutta_4(@(t, X) chua_system(t, X, alpha, beta, gamma, m0, m1), tspan, x0, Ts);

% Creating NARX dataset
delay = 2;
X_narx = X(1:end-delay, :); % Inputs
Y_narx = X(delay+1:end, :); % Outputs


num_neurons = 15; % Number of neurons
model = elm_train(X_narx, Y_narx, num_neurons); % ELM Model training

Y_pred = elm_predict(X_narx, model); % ELM Model Prediction


figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
gif('ChuaNARXID.gif')
for k = 1:length(Y_narx)
    if mod(k,100)==0
        subplot(221);
        plot((1:k)*Ts,Y_narx(1:k,1), 'k', 'LineWidth', 2); hold on; plot((1:k)*Ts,Y_pred(1:k,1), 'r--', 'LineWidth', 2); hold off;
        legend('Actual', 'Prediction');title('State x');xlabel("Time [Sn]")

        subplot(222);
        plot((1:k)*Ts,Y_narx(1:k,2), 'k', 'LineWidth', 2); hold on; plot((1:k)*Ts,Y_pred(1:k,2), 'r--', 'LineWidth', 2); hold off;
        legend('Actual', 'Prediction');title('State y');;xlabel("Time [Sn]")

        subplot(223);
        plot((1:k)*Ts,Y_narx(1:k,3), 'k', 'LineWidth', 2); hold on; plot((1:k)*Ts,Y_pred(1:k,3), 'r--', 'LineWidth', 2); hold off;
        legend('Actual', 'Prediction');title('State z');;xlabel("Time [Sn]")

        % Phase space plot
        subplot(224);
        plot3(X(1:k,1), X(1:k,2), X(1:k,3), 'b'); hold on; % Original system
        plot3(Y_pred(1:k,1), Y_pred(1:k,2), Y_pred(1:k,3), 'r--', 'LineWidth', 2); hold off; % Predicted system
        legend('Original Phase Space', 'Predicted Phase Space');
        xlabel('State x'); ylabel('State y');zlabel("State z")
        title(["Phase Space";"||Error|| : "+num2str(norm(X(1:k)-Y_pred(1:k)))]);
        view(45, 20)
        drawnow
        gif
    end
end

% Chua circuit system function
function dXdt = chua_system(~, X, alpha, beta, gamma, m0, m1)
    x = X(1);
    y = X(2);
    z = X(3);

    % Chua's nonlinear characteristic
    f_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1));

    % Differential equations
    dxdt = alpha * (y - x - f_x);
    dydt = x - y + z;
    dzdt = -beta * y - gamma * z;

    dXdt = [dxdt; dydt; dzdt];
end

function [t, X] = runge_kutta_4(odefun, tspan, x0, Ts)
    t0 = tspan(1);
    tf = tspan(2);
    t = t0:Ts:tf;
    n = length(t);
    m = length(x0);
    X = zeros(n, m);
    X(1, :) = x0;

    for i = 1:n-1
        k1 = Ts * odefun(t(i), X(i, :)');
        k2 = Ts * odefun(t(i) + 0.5*Ts, X(i, :)' + 0.5*k1);
        k3 = Ts * odefun(t(i) + 0.5*Ts, X(i, :)' + 0.5*k2);
        k4 = Ts * odefun(t(i) + Ts, X(i, :)' + k3);
        X(i+1, :) = X(i, :) + (1/6) * (k1 + 2*k2 + 2*k3 + k4)';
    end
end

function model = elm_train(X, Y, num_neurons)
    [num_samples, num_inputs] = size(X);
    [num_samples, num_outputs] = size(Y);

    input_weights = rand(num_neurons, num_inputs) * 2 - 1;
    bias = rand(num_neurons, 1);
    H = 1 ./ (1 + exp(-(X * input_weights' + bias')));
    output_weights = pinv(H) * Y;
    model.input_weights = input_weights;
    model.bias = bias;
    model.output_weights = output_weights;
end

function Y_pred = elm_predict(X, model)
    H = 1 ./ (1 + exp(-(X * model.input_weights' + model.bias')));
    Y_pred = H * model.output_weights;
end
