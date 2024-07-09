clc; clear; close all

%% Training Data
% We genreate ns training samples using ode45 in a loop
% At the begining of each loop, a random IC is being generated
% Given the IC, ode45 solves the system for tf amount of time
% The solution at the finial time step tf is extracted as the next state

g = 9.81; 
l = 1; 

tf = 0.1;
tspan = [0 tf];
% dt = 0.001;
% tspan = 0:dt:tf; 

f = @(t, x) [x(2); -(g/l)*sin(x(1))];
ode_options = odeset(RelTol=1e-10, AbsTol=1e-11);


x_tot = [];
y_tot = [];
ns = 15000;
for i = 1:ns
    x0 = [pi*rand-pi/2 10*rand-5];
    [~, x_out] = ode45(f, tspan, x0);
    x_tot = [x_tot; x0];
    y_tot = [y_tot; x_out(end,:)];
end
x_train = x_tot;
y_train = y_tot;

% x_data = x_tot;
% y_data = y_tot;

train_data_file_name = 'Pendulum_train_data_tf_0_1_ns_15000.mat';
% save(train_data_file_name, 'x_train', 'y_train')
% save(train_data_file_name, 'x_data', 'y_data')


%% Testing Data
% Given an initial condition, ode45 solves the system for the next state
% The computed state is then being fed back to the ode45 to solve for
% this process continues untill nt steps of the system

x0 = [pi/3 1]; 

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

test_data_file_name = 'Pendulum_test_data_tf_0_1_nt_1000.mat';
% save(test_data_file_name, 'x_test', 'y_test')

clear x_tot y_tot

plot(y_test(:,1))
hold on
plot(x_test(:,1))

%% Full ODE
% For the one-step ode solution, I had to use the ode_options in order to
% increase the accuracy of the ode solver so that it better matches the
% ode45 in loop solution

tspan = 0:tf:nt*tf; 
[~, x_out] = ode45(f, tspan, x0, ode_options);

plot(x_out(:,1),'b--',LineWidth=2)
legend('y_{test}','x_{test}','ode')
grid on