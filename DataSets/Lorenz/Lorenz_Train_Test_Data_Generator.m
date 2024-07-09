clc; clear; close all

%% Training Data
% We genreate ns training samples using ode45 in a loop
% At the begining of each loop, a random IC is beinf generated
% Given the IC, ode45 solves the system for tf amount of time
% The solution at the finial time step tf is extracted as the next state

% system parameters
sigma = 10;
beta = 8/3;
rho = 28;

tf = 1e-2;      % final time
tspan = [0 tf];
% dt = 1e-4;
% tspan = 0:dt:tf; 

% Lorenz Model
f = @(t, x) [sigma*(x(2)-x(1));...
             rho*x(1) - x(1)*x(3) - x(2); ...
             x(1)*x(2) - beta*x(3)];
ode_options = odeset(RelTol=1e-10, AbsTol=1e-11);

x_tot = [];
y_tot = [];
ns = 60000;     % number of training samples
for i = 1:ns
    x0 = 30*(rand(1,3)-0.5);
    [~, x_out] = ode45(f, tspan, x0);
    x_tot = [x_tot; x0];
    y_tot = [y_tot; x_out(end,:)];
end
% x_train = x_tot;
% y_train = y_tot;

x_data = x_tot;
y_data = y_tot;

train_data_file_name = 'lorenz_train_data_tf_0_01_ns_60000.mat';
% save(train_data_file_name, 'x_train', 'y_train')
save(train_data_file_name, 'x_data', 'y_data')


%% Testing Data
% Given an initial condition, ode45 solves the system for the next state
% The computed state is then being fed back to the ode45 to solve for
% this process continues untill nt steps of the system

x0 = 20*(rand(1,3)-0.5); 
x0 = [8.4553    3.2408    9.1099];

nt = 1000;
x_tot = [];
y_tot = [];

x_tot(1,:) = x0;
for i = 1:nt
    x0_temp = x_tot(i,:);
    [~, x_out] = ode45(f, tspan, x0_temp);
    if i ~= nt
        x_tot(i+1,:) = x_out(end,:);
    end
    y_tot(i,:) = x_out(end,:);
end
x_test = x_tot;
y_test= y_tot;

test_data_file_name = 'lorenz_test_data_tf_0_01_nt_1000.mat';
% save(test_data_file_name, 'x_test', 'y_test')

clear x_tot y_tot

%% Full ODE
% For the one-step ode solution, I had to use the ode_options in order to
% increase the accuracy of the ode solver so that it better matches the
% ode45 in loop solution

tspan = 0:tf:nt*tf; 
[~, x_out] = ode45(f, tspan, x0, ode_options);

%% Plots 
figure(1)
plot3(x0(1), x0(2), x0(3), 'ro', 'LineWidth',2)
hold on
plot3(y_test(:,1), y_test(:,2), y_test(:,3),'g')
plot3(x_out(:,1), x_out(:,2), x_out(:,3),'b--')
legend('IC','ode45 (in loop)', 'ode45 (one-step)',Location='best')
grid on
view(-15,30)


figure(2)
subplot(3,1,1)
plot(y_test(:,1),'g',LineWidth=2)
hold on 
plot(x_out(2:end,1),'b--',LineWidth=2)
legend('ode45 (in loop)', 'ode45 (one-step)',Location='best')

subplot(3,1,2)
plot(y_test(:,2),'g',LineWidth=2)
hold on 
plot(x_out(2:end,2),'b--',LineWidth=2)
legend('ode45 (in loop)', 'ode45 (one-step)',Location='best')

subplot(3,1,3)
plot(y_test(:,3),'g',LineWidth=2)
hold on 
plot(x_out(2:end,3),'b--',LineWidth=2)
legend('ode45 (in loop)', 'ode45 (one-step)',Location='best')
