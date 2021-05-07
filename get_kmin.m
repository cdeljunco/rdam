%simulation parameters
%Da = 0.2
Da = 0.4
%Da = 0.8;
Omegalist = [linspace(1, 10, 10)];

k_fullsim_list = ones(size(Omegalist))*2/sqrt(Da);

%k_fullsim_list = csvread(strcat('kmin_extensile_da', sprintf('%.1f', Da), '.csv')); %k values from full simulation

kmin = zeros(size(Omegalist));

%ode integration parameters
tspan = [0 500];
x0 = [1 -1e-7];
options = odeset('MaxStep', 1);

%set to 1 to show x vs t -- WARNING: THIS WILL MAKE 40*length(Omegalist) FIGURES
show_traj = 0;

%cycle through values of Omega
figure
j = 1;
for Omega = Omegalist
    i = 1;
    k_fullsim = k_fullsim_list(j);
    %use values from full simulation to guess possible range of front speed values  
    klist = k_fullsim - 0.2 : 0.01 : k_fullsim + 0.2;
    xmin = zeros(size(klist));
    for k = klist
        %integrate ODE
        [t,x] = ode23s(@(t,x) particle(t, x, Da, Omega, k), tspan, x0, options);
        if show_traj==1
            figure
            plot(t, x(:, 1), '-o');
        end
        %get minimum value of x in trajectory
        xmin(i) = abs(min(x(:,1)));
        i = i + 1;
    end
    %plot front speed vs min abs value of x
    semilogy(klist, xmin, '-o')
    hold on
    %get value of k where kmin first drops to ~ 0; this is where the value
    %of xmin changes fastest as a function of k
    deltaxmin = abs(log(abs(xmin(2:end) - xmin(1:length(xmin)-1))));
    [m, index] = max(deltaxmin);
    kmin(j) = klist(index - 1);
    j = j + 1;
end
ylim([10^-50 1])
xlabel('k')
ylabel('minimum value of x')
legendCell = cellstr(num2str(Omegalist', '%-d'));
legend(legendCell)
hold off
%plot Omega vs min k
figure
plot(-Omegalist, kmin, '--or')
hold on
plot(-Omegalist, k_fullsim_list(1:length(Omegalist)), '--ob')
legend('ode simulation','full simulation')
xlabel('Omega')
ylabel('minimum value of k')
csvwrite(strcat('kmin_nm_da', sprintf('%.1f', Da), '.csv'), kmin)


        
