clc,clear,close all;
% LTI System ARX Identification via RLSE
% Written By: Rasit
% 13-Dec-2024
%%
Ts = 1e-3;     % Sampling Periode
SimTime = linspace(0,10,5/Ts)';  % Sim Time
NoD = size(SimTime,1);   % Number Of Data
sys = tf(4,[1 2 4]);  % Transfer Function
sysd = c2d(sys,Ts);    % System Discrete Transfer Function
num = cell2mat(sysd.Numerator);     % zeros
den = -cell2mat(sysd.Denominator);   % poles
RealParam = [den(2) den(3) num(2) num(3)];   % Real Model Parameter


%% Discrete Transfer Function Simulation
u =  ones(NoD,1);   % System Input
y =  zeros(NoD,1);  % System Output

for i=size(den,2):size(SimTime,1)
    y(i) = den(2)*y(i-1) + den(3)*y(i-2) + num(2)*u(i-1) + num(3)*u(i-2);
end

%% Recursive (Online) System Identification
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
rng(10)
NoP = size(RealParam,2);  % Number Of Param
b = y;
Fi = zeros(NoD,NoP);
xRlse = 0*rand(NoP,1);     % Random start RLSE state vector
xRlseHorizon = zeros(NoP,NoD);
P = 1e12*eye(NoP,NoP);    % covariance matrix
for i=size(den,2):size(y,1)
    % ARX Model
    Fi(i,:)  = [y(i-1) y(i-2) u(i-1) u(i-2)];     % Regressor vector
    b = y(i);  % Measurement

    [xRlse,K,P] = rlse_online(Fi(i,:),b,xRlse,P);
    xRlseHorizon(:,i) = xRlse;
    yID = zeros(NoD,1);
    for j=size(den,2):size(SimTime,1)
        yID(j) = xRlse(1)*yID(j-1) + xRlse(2)*yID(j-2) + xRlse(3)*u(j-1) + xRlse(4)*u(j-2);
    end
    if mod(i,1e2) == 0
        clf
        subplot(121)
        plot(SimTime,u,"r",LineWidth=4),hold on,grid on
        plot(SimTime,y,"b",LineWidth=4)
        plot(SimTime,yID,"k--",LineWidth=4)
        title("Time Response")
        legend("Input Signal","Real System Response","SysID Response")

        subplot(122)
        plot(ones(NoD,NoP).*RealParam,"r",LineWidth=5),hold on,grid on
        plot(ones(NoD,NoP).*xRlseHorizon',"k.",LineWidth=3)
        title("Parameters")
        drawnow
    end
end


disp("Poles | Poles | Zeros | Zeros")
disp("Real Model Parameters = "+num2str(RealParam))
disp("-----------------------")
disp("Estimated Model Parameters = "+num2str(xRlse'))
disp("-----------------------")


function [x,K,P] = rlse_online(ak,bk,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
ak = ak(:);
bk = bk(:);
K = (P*ak)/(ak'*P*ak+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(bk-ak'*x);     % State Update
P = P - K*ak'*P;           % Covariance Update
end


