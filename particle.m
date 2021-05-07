function dxdt = particle(t, x, Da, Omega, k)

%Integrate the ODE resulting from assuming l = 0 + traveling wave approximation

%x(1) == x
%x(2) == x'

%x' = x(2)
%x'' = (((k - Omega * x') * x') + (x * (1 - x))) / ((1/Da) - Omega * x)
%    = (((k - Omega * x(2)) * x(2)) + (x(1) * (1 - x(1))))/( (1/Da) - Omega * x(1) )

dxdt =[x(2); -(((k - Omega * x(2)) * x(2)) + (x(1) * (1 - x(1)))) / ((1/Da) - Omega * x(1))];