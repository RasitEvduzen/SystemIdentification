clc,clear,close all;
% LTI System ARX Identification via LSE
% Written By: Rasit
% 13-Dec-2024
%%
Ts = 1e-2;     % Sampling Periode
SimTime = linspace(0,5,5/Ts)';  % Sim Time
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
%% Batch (Offline) System Identification
NoP = size(RealParam,2);  % Number Of Param
b = y;
Fi = zeros(NoD,NoP);
for i=size(den,2):size(y,1)
    % ARX Model
    Fi(i,:)  = [y(i-1) y(i-2) u(i-1) u(i-2)];     % Regressor vector
end
xlse = pinv(Fi'*Fi)*Fi'*b;  % LSE Param

yID = zeros(NoD,1);
for i=size(den,2):size(SimTime,1)
    yID(i) = xlse(1)*yID(i-1) + xlse(2)*yID(i-2) + xlse(3)*u(i-1) + xlse(4)*u(i-2);
end


%% Plot Sim
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')

plot(SimTime,u,"r",LineWidth=3),hold on,grid on
plot(SimTime,y,"b",LineWidth=3)
plot(SimTime,yID,"k--",LineWidth=3)

legend("Input Signal","Real System Response","SysID Response")
disp("-----------------------")
disp("Poles | Poles | Zeros | Zeros")
disp("Real Model Parameters = "+num2str(RealParam))
disp("-----------------------")
disp("Estimated Model Parameters = "+num2str(xlse'))
disp("-----------------------")




