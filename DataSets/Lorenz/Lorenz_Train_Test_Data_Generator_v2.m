clc; clear; close all

%% Training Data
% We genreate ns training samples using ode45 in a loop
% At the beginning of each loop, a random IC is being generated
% Given the IC, ode45 solves the system for 0:dt:T amount of time
% The data 1:end-1 of the solution of each trajectory -> x_tot
% The data 2:end of the solution of each trajectory -> y_tot

% system parameters
sigma = 10;
beta = 8/3;
rho = 28;

dt = 1e-2;      % final time
T = 10;
t = 0:dt:T;

% Lorenz Model
f = @(t, x) [sigma*(x(2)-x(1));...
             rho*x(1) - x(1)*x(3) - x(2); ...
             x(1)*x(2) - beta*x(3)];

ode_options = odeset(RelTol=1e-10, AbsTol=1e-11);

x_tot = [];
y_tot = [];
nst = 100;     % number of training sample trajectories

for i = 1:nst
    x0 = 30*(rand(1,3)-0.5);
    [~, x_out] = ode45(f, t, x0);
    x_tot = [x_tot; x_out(1:end-1, :)];
    y_tot = [y_tot; x_out(2:end , :)];
    plot3(x_out(:,1), x_out(:,2), x_out(:,3));
    hold on
    plot3(x0(1), x0(2), x0(3),'ro');
end
grid on
view(-23, 18)

% x_train = x_tot;
% y_train = y_tot;

x_data = x_tot;
y_data = y_tot;

train_data_file_name = 'lorenz_train_data_dt_0_01_nst_100_T_10.mat';
% save(train_data_file_name, 'x_train', 'y_train')
% save(train_data_file_name, 'x_data', 'y_data')

clear x_tot y_tot

%% Test Data
t = 0:dt:30;
x0 = 20*(rand(1,3)-0.5); 

[~, x_out] = ode45(f, t, x0);
x_test = x_out(1:end-1, :);
y_test = x_out(2:end , :);

test_data_file_name = 'lorenz_test_data_dt_0_01_T_30.mat';
% save(test_data_file_name, 'x_test', 'y_test')

clear x_out
%% Plots 
figure
plot3(x0(1), x0(2), x0(3), 'ro', 'LineWidth',2)
hold on
plot3(y_test(:,1), y_test(:,2), y_test(:,3),'b')
grid on
view(-15,30)