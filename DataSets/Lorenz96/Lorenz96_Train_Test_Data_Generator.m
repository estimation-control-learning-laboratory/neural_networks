clc; clear; close all;

%% Training Data
% We generate ns training samples using ode45 in a loop
% At the beginning of each loop, a random IC is being generated
% Given the IC, ode45 solves the system for 0:dt:T amount of time
% The data 1:end-1 of the solution of each trajectory -> x_tot
% The data 2:end of the solution of each trajectory -> y_tot

% Parameters for the Lorenz 96 system
N = 5;
F = 8;

dt = 1e-2;      % final time
T = 10;
t = 0:dt:T;

% Lorenz 96 model
Lorenz96 = @(t, x) arrayfun(@(i) (x(mod(i, N) + 1) - x(mod(i-2, N) + 1)) * x(mod(i-1, N) + 1) - x(i) + F, 1:N)';

% ode_options = odeset('RelTol', 1e-10, 'AbsTol', 1e-11);

x_tot = [];
y_tot = [];
nst = 100;  % Number of training sample trajectories

for i = 1:nst
    x0 = F * ones(1, N) + 10*(rand(1, N)-0.5);  % Initial condition
    [~, x_out] = ode45(Lorenz96, t, x0);
    x_tot = [x_tot; x_out(1:end-1, :)];
    y_tot = [y_tot; x_out(2:end , :)];
end

x_data = x_tot;
y_data = y_tot;

% train_data_file_name = 'lorenz96_train_data_dt_0_01_nst_100_T_10.mat';
% save(train_data_file_name, 'x_data', 'y_data');

clear x_tot y_tot;

%% Test Data

x0 = F * ones(1, N) + 20*(rand(1, N)-0.5);  % Initial condition for test

[~, x_out] = ode45(Lorenz96, t, x0);
x_test = x_out(1:end-1, :);
y_test = x_out(2:end , :);

% test_data_file_name = 'lorenz96_test_data_dt_0_01_T_10.mat';
% save(test_data_file_name, 'x_test', 'y_test');

figure;
for i = 1:N
    subplot(N, 1, i);
    plot(t(1:end-1), x_test(:, i), LineWidth=2);
    title(['State x_', num2str(i)]);
end
set(gcf,'position',[200,100,1400,800])

clear x_out;