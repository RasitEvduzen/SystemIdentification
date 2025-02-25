function [ x1, x2, x3, y ] = CSTR( x1, x2, x3, u, Ts )
% CSTR - Continuous Stirred Tank Reactor Model using Runge-Kutta Integration

x1ini = x1;
x2ini = x2;
x3ini = x3;

K11 = Ts * (1 - 4*x1 + 0.5*x2^2);
K21 = Ts * (-x2 + 3*x1 - 1.5*x2^2 + u);
K31 = Ts * (-x3 + x2^2);

x1 = x1ini + K11/2;
x2 = x2ini + K21/2;
x3 = x3ini + K31/2;

K12 = Ts * (1 - 4*x1 + 0.5*x2^2);
K22 = Ts * (-x2 + 3*x1 - 1.5*x2^2 + u);
K32 = Ts * (-x3 + x2^2);

x1 = x1ini + K12/2;
x2 = x2ini + K22/2;
x3 = x3ini + K32/2;

K13 = Ts * (1 - 4*x1 + 0.5*x2^2);
K23 = Ts * (-x2 + 3*x1 - 1.5*x2^2 + u);
K33 = Ts * (-x3 + x2^2);

x1 = x1ini + K13;
x2 = x2ini + K23;
x3 = x3ini + K33;

K14 = Ts * (1 - 4*x1 + 0.5*x2^2);
K24 = Ts * (-x2 + 3*x1 - 1.5*x2^2 + u);
K34 = Ts * (-x3 + x2^2);

x1 = x1ini + (K11/6 + K12/3 + K13/3 + K14/6);
x2 = x2ini + (K21/6 + K22/3 + K23/3 + K24/6);
x3 = x3ini + (K31/6 + K32/3 + K33/3 + K34/6);

y = x3; 

end
