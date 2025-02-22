clc; clear all; close all;
% NARX System Identification
% Written By: Rasit
% Date: 22-Feb-2025
%% MISO Neural Network Training 
load CSTR.mat

T = INPUT;   % Input dataset
Y = OUTPUT;  % Output dataset

%% Neural Network Model Setup
[N,R] = size(T);  % N = Number of total data points, R = Number of input variables
S = 15;           % Number of neurons in the hidden layer

% Initialize random weights and biases
Wg = rand(S,R) - 0.5;   % Input weight matrix (random initialization between -0.5 and 0.5)
bh = rand(S,1) - 0.5;   % Hidden layer biases
Wc = rand(1,S) - 0.5;   % Output weight matrix
bc = rand(1,1) - 0.5;   % Output bias

%% Split Data into Training and Validation Sets
TrainingIndex   = 1:2:N;
ValidationIndex = 2:2:N;
TrainingINPUT   = T(TrainingIndex,:);
TrainingOUTPUT  = Y(TrainingIndex,:);
ValidationINPUT = T(ValidationIndex,:);
ValidationOUTPUT = Y(ValidationIndex,:);
Ntraining   = size(TrainingINPUT,1);
Nvalidation = size(ValidationINPUT,1);

% Ensure the number of parameters does not exceed the training data count
if S*(R+2)+1 > Ntraining    
    disp('Too many neurons! Reduce S.');
    return
end

%% Levenberg-Marquardt Training
Nmax = 50; % Maximum number of iterations
I = eye(S*(R+2)+1);
condition = 1; 
iteration = 0; 
mu = 1; 
FvalMIN = inf;

while condition
    iteration = iteration + 1;
    [ yhat ] = MISOYSAmodel(TrainingINPUT, Wg, bh, Wc, bc);
    eTra = TrainingOUTPUT - yhat;                               
    f = eTra' * eTra; % Compute error

    % Compute Jacobian matrix
    J = [];
    for i = 1:Ntraining
        for j = S*(R+2)+1
            J(i,j) = -1;
        end
        for j = S*(R+1)+1:S*(R+2)
            J(i,j) = -tanh(Wg(j-(R+1)*S,:) * TrainingINPUT(i,:)' + bh(j-(R+1)*S));
        end
        for j = S*R+1:S*R+S
            J(i,j) = -Wc(1, j-S*R) * (1 - tanh(Wg(j-S*R,:) * TrainingINPUT(i,:)' + bh(j-S*R))^2);
        end
        for j = 1:S*R
            k = mod(j-1, S) + 1;
            m = fix((j-1) / S) + 1;
            J(i,j) = -Wc(1,k) * TrainingINPUT(i,m) * (1 - tanh(Wg(k,:) * TrainingINPUT(i,:)' + bh(k))^2);
        end
    end
    
    loop2 = 1;
    while loop2
        p = -inv(J' * J + mu * I) * J' * eTra;
        [x] = matrix2vector(Wg, bh, Wc, bc);
        [Wgz, bhz, Wcz, bcz] = vector2matrix(x + p, S, R);
        [yhatz] = MISOYSAmodel(TrainingINPUT, Wgz, bhz, Wcz, bcz);
        fz = (TrainingOUTPUT - yhatz)' * (TrainingOUTPUT - yhatz);

        if fz < f
            x = x + p;
            [Wg, bh, Wc, bc] = vector2matrix(x, S, R);
            mu = 0.1 * mu;
            loop2 = 0;
        else
            mu = 10 * mu;
            if mu > 1e+20
                loop2 = 0;
                condition = 0;
            end
        end
    end
    
    [ yhat ] = MISOYSAmodel(TrainingINPUT, Wg, bh, Wc, bc);
    eTra = TrainingOUTPUT - yhat;
    f = eTra' * eTra;
    FTRAINING(iteration) = f;
    
    [yhat] = MISOYSAmodel(ValidationINPUT, Wg, bh, Wc, bc);
    eVal = ValidationOUTPUT - yhat;
    fVALIDATION = eVal' * eVal;
    FVALIDATION(iteration) = fVALIDATION;

    if fVALIDATION < FvalMIN
        xbest = x;
        FvalMIN = fVALIDATION;
    end

    g = 2 * J' * eTra;
    fprintf('Iteration: %4.0f ||g||: %4.6f f(x): %4.6f fv: %4.6f\n', iteration, norm(g), f, fVALIDATION);

    if iteration >= Nmax
        condition = 0;
    end
end

%% Plot Results
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')

[Wg, bh, Wc, bc] = vector2matrix(xbest, S, R);
[yhatTR] = MISOYSAmodel(TrainingINPUT, Wg, bh, Wc, bc);
[yhatVA] = MISOYSAmodel(ValidationINPUT, Wg, bh, Wc, bc);

subplot(2,2,1)
plot(TrainingIndex, TrainingOUTPUT, 'r',LineWidth=2);
hold on, grid
plot(TrainingIndex, yhatTR, 'k--',LineWidth=2);
legend('Training Data', 'Training Output')
title("Model Training Phase")

subplot(2,2,2)
plot(ValidationIndex, ValidationOUTPUT, 'r',LineWidth=2);
hold on, grid
plot(ValidationIndex, yhatVA, 'k--',LineWidth=2);
legend('Test Data', 'Test Output')
title("Model Test Phase")

subplot(2,2,[3,4])
plot(FTRAINING, 'b',LineWidth=2)
hold on, grid
plot(FVALIDATION, 'r--',LineWidth=2)
legend('Training Error', 'Validation Error')
title(["Number of Neurons: "+ num2str(S);"Best Validation Error: "+ num2str(FvalMIN)])

%% Utility Functions
function [ Wg, bh, Wc, bc ] = vector2matrix( x, S, R )
% vector2matrix - Converts a Parameter Vector Back into Matrices and Biases
%
% This function reconstructs the weight matrices and bias vectors of a neural network
% from a single-column vector (typically used in optimization or storage).
%
% Inputs:
%   x - A single-column vector containing all the parameters.
%   S - Number of hidden layer neurons (size of Wg rows).
%   R - Number of input features (size of Wg columns).
%
% Outputs:
%   Wg - Weight matrix for the hidden layer (S x R).
%   bh - Bias vector for the hidden layer (S x 1).
%   Wc - Weight matrix for the output layer (1 x S).
%   bc - Bias for the output layer (scalar).

Wg = [];  

% Extract Wg (hidden layer weights)
for r = 1:R
    Wg = [Wg, x((r-1)*S+1 : r*S)];
end

% Extract hidden layer bias (bh)
bh(:,1) = x(S*R+1 : S*R+S);

% Extract output layer weights (Wc)
Wc(1,:) = x(S*(R+1)+1 : S*(R+2));

% Extract output layer bias (bc)
bc = x(S*(R+2)+1);

end


function [ x ] = matrix2vector( Wg, bh, Wc, bc )
% matrix2vector - Converts Neural Network Weights and Biases into a Single Vector
%
% This function reshapes the weight matrices and bias vectors of a neural network
% into a single column vector for optimization or storage purposes.
%
% Inputs:
%   Wg - Weight matrix for the hidden layer.
%   bh - Bias vector for the hidden layer.
%   Wc - Weight matrix for the output layer.
%   bc - Bias for the output layer.
%
% Output:
%   x  - A single-column vector containing all the parameters.
  
R = size(Wg, 2);  % Number of input features

x = [];  

% Flatten Wg column-wise and append to x
for r = 1:R
    x = [x; Wg(:, r)];
end

% Append the bias and output layer weights
x = [x; bh];   % Append hidden layer bias
x = [x; Wc'];  % Append output layer weights (transposed)
x = [x; bc];   % Append output layer bias

end



function [ yhat ] = MISOYSAmodel( T, Wg, bh, Wc, bc )
% MISOYSAmodel - Multi-Input Single-Output (MISO) Neural Network Model
% 
% This function computes the estimated output (yhat) for a given input matrix T
% using a simple neural network model with hyperbolic tangent activation.
%
% Inputs:
%   T  - Input data matrix (N x R), where N is the number of data points, and R is the number of inputs.
%   Wg - Weight matrix for the hidden layer.
%   bh - Bias vector for the hidden layer.
%   Wc - Weight matrix for the output layer.
%   bc - Bias for the output layer.
%
% Output:
%   yhat - Estimated output vector (N x 1).

[N, R] = size(T); % N: Number of data points, R: Number of input features

% Process each data point
for n = 1:N
    yhat(n,1) = Wc * tanh(Wg * T(n,:)' + bh) + bc; % Compute output using activation function
end

end