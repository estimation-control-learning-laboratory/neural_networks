clc; clear; close all

%% Training Data
% We generate ns training samples using ode45 in a loop
% At the beginning of each loop, a random IC is being generated
% Given the IC, ode45 solves the system for 0:dt:T amount of time
% The data 1:end-1 of the solution of each trajectory -> x_tot
% The data 2:end of the solution of each trajectory -> y_tot

% system parameters
m1 = 1;
m2 = 0.3;
ell_1 = 0.5;
ell_2 = 0.5;
g = 10;

dt = 1e-2;      % time step
T = 10;
t = 0:dt:T;

% Dynamics function for the double pendulum
syms theta1 theta2 theta1dot theta2dot real
phi = theta2 - theta1;

M = [(m1 + m2) * ell_1^2, m2 * ell_1 * ell_2 * cos(phi); 
     ell_1 * cos(phi), ell_2];

RHS = [0; 0] - ...
      [-m2 * ell_1 * ell_2 * sin(phi) * theta2dot^2 + (m1 + m2) * g * ell_1 * sin(theta1); 
       ell_1 * sin(phi) * theta1dot^2 + g * sin(theta2)];

f = [theta1dot; theta2dot; M \ RHS];
F = [f(1); f(3); f(2); f(4)];
dynamicsFunction = matlabFunction(F, 'Vars', [theta1, theta2, theta1dot, theta2dot]);

ode_options = odeset('RelTol', 1e-10, 'AbsTol', 1e-11);

x_tot = [];
y_tot = [];
nst = 500;     % number of training sample trajectories

for i = 1:nst
    % Random initial conditions within a range
    x0 = [pi * (rand - 0.5); pi * (rand - 0.5); (rand - 0.5); (rand - 0.5)];
    [~, x_out] = ode45(@(t, x) DoublePendulumDynamics(x, dynamicsFunction), t, x0, ode_options);
    x_tot = [x_tot; x_out(1:end-1, :)];
    y_tot = [y_tot; x_out(2:end, :)];
end


x_data = x_tot;
y_data = y_tot;

train_data_file_name = 'double_pendulum_train_data_dt_0_01_nst_500_T_10.mat';
% save(train_data_file_name, 'x_data', 'y_data')

clear x_tot y_tot

%% Test Data
t = 0:dt:10;
% x0 = [pi * (rand - 0.5); pi * (rand - 0.5); (rand - 0.5); (rand - 0.5)];
x0 = [pi/4; pi/4; 0; 0];

[~, x_out] = ode45(@(t, x) DoublePendulumDynamics(x, dynamicsFunction), t, x0, ode_options);
x_test = x_out(1:end-1, :);
y_test = x_out(2:end, :);

test_data_file_name = 'double_pendulum_test_data_dt_0_01_T_10.mat';
% save(test_data_file_name, 'x_test', 'y_test')

clear x_out

%% Plots
figure
set(gcf, 'position', [200, 100, 800, 700])

subplot(4,1,1)
plot(y_test(:, 1), 'LineWidth', 2);
grid on 
axis tight 

subplot(4,1,2)
plot(y_test(:, 2), 'LineWidth', 2);
grid on 
axis tight 

subplot(4,1,3)
plot(y_test(:, 3), 'LineWidth', 2);
grid on 
axis tight 

subplot(4,1,4)
plot(y_test(:, 4), 'LineWidth', 2);
grid on 
axis tight 

%% Functions
function x_next = DoublePendulumDynamics(x, dynamicsFunction)
    % Dynamics of the double pendulum
    x_next = dynamicsFunction(x(1), x(2), x(3), x(4));
end
