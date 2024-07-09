clc; clear; close all

% Loading the training and the test data sets
addpath DataSets\Pendulum\
addpath NN_Functions\
load('Pendulum_train_data_tf_0_1_ns_15000.mat')
load('Pendulum_test_data_tf_0_1_nt_500.mat')

input_size  = size(x_data,2); 
output_size = size(y_data,2); 
hidden_size = 10;

% Define network architecture
layers = [
    featureInputLayer(input_size, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(hidden_size, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(output_size, 'Name', 'fc2')
];

lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

% Train, Validation and Test Data
train_idx = floor(size(x_data,1) * 0.8);

train_inputs  = x_data(1:train_idx, :);
train_targets = y_data(1:train_idx, :);

val_inputs  = x_data(train_idx+1:end, :);
val_targets = y_data(train_idx+1:end, :);

test_inputs  = x_test;
test_targets = y_test;

% Training options
epochs = 100;
batch_size = 32;
learning_rate_init = 0.001;
validation_freq = 30;

% Training the Model
% [dlnet,num_iters,trainLoss,valLoss] = train_net_adam(dlnet, train_inputs, train_targets, val_inputs, val_targets, ...
%     epochs, batch_size, learning_rate_init, validation_freq);
% 
% save('Pendulum_trained_network_dlnet.mat', 'dlnet')
% save('Pendulum_train_val_loss_dlnet.mat', 'trainLoss', 'valLoss', 'num_iters')

% Using a trained model
load("Pendulum_trained_network_dlnet.mat")
load("Pendulum_train_val_loss_dlnet.mat")

%% Model Prediction Test
% Convert test inputs to dlarray
dlTestX = dlarray(test_inputs', 'CB');

% Predict using the trained network
dlPredictedOutputs = predict(dlnet, dlTestX);
predicted_outputs = gather(extractdata(dlPredictedOutputs))';

% Plot the results
figure
set(gcf, 'Position', [200, 100, 1000, 700])

subplot(2, 1, 1)
set(gca,'fontsize',16);
plot(test_targets(:, 1), 'g', 'LineWidth', 2)
hold on
plot(predicted_outputs(:, 1), 'b--', 'LineWidth', 2)
ylabel('$x_1$', Interpreter='latex', fontsize=22)
grid on
legend('$x_1$', '$\hat x_1$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=18)
title('Predictions')

subplot(2, 1, 2)
set(gca,'fontsize',16);
plot(test_targets(:, 2), 'g', 'LineWidth', 2)
hold on
plot(predicted_outputs(:, 2), 'b--', 'LineWidth', 2)
ylabel('$x_2$', Interpreter='latex', fontsize=22)
grid on
legend('$x_2$', '$\hat x_2$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=18)

%% EKF Augmentation

% Initial states
x_dim = input_size;
num_samples = size(test_inputs, 1);
x_est = test_inputs(1, :)';
P = 1e-4 * eye(x_dim);

% Process and measurement noise
process_var = 0.001;
Q = process_var * eye(x_dim);   
measurement_var = 0.01;
R = measurement_var;

% Initilization
x_est_store = zeros(x_dim, num_samples);
C = [1 0];
y_tot = zeros(size(C,1), num_samples);
estimation_error = zeros(x_dim, num_samples);
estimation_error_norm_squared = zeros(1, num_samples);
J_cost = zeros(1, num_samples);
eigs = zeros(x_dim, num_samples);


% Estimation using EKF
count = 0;
for k = 1:num_samples
    inputData = dlarray(x_est, 'CB');
    [output, jacobian] = dlfeval(@computeJacobian, dlnet, inputData);
    A = jacobian;

    % [output, J_analytical] = computeJacobian_Analytical(dlnet, x_est);
    % A = J_analytical;

    eigs(:,k) = eig(A); 
    if norm(eigs(1,k))<1 && norm(eigs(2,k))<1
        count = count + 1;
    end

    dlNextState = predict(dlnet, inputData);    % Prior estimate
    x_est = gather(extractdata(dlNextState));

    [K, P, J_cost(k)] = Compute_KPJ_EKF(A, C, Q, R, P);      % EKF: Prior & Posterior P and K caculation

    y = C*test_targets(k,:)' + sqrt(measurement_var)*randn;    % Measurement
    y_tot(k) = y;

    % x_est = x_est + K*(y - C*x_est);               % Posterior estimate

    estimation_error(:, k) = abs(x_est - test_targets(k,:)');   % Estimation Error
    estimation_error_norm_squared(:, k) = norm(x_est - test_targets(k,:)')^2; % Estimation Error Norm Squared

    x_est_store(:, k) = x_est;
end

%% Plots

figure
set(gcf,'position',[200,100,800,700])
subplot(3,1,1)
plot(test_targets(:,1),'k', LineWidth=2)
hold on
plot(x_est_store(1,:),'b--', LineWidth=2)
set(gca,'fontsize',16);
set(gca,'xticklabel',{[]})
ylabel('$x_1$', Interpreter='latex', fontsize=22)
grid on 
ylim([-1.5 2])
legend('$x_{1,k}$', '$\hat x_{1,k|k}$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=18)

subplot(3,1,2)
plot(test_targets(:,2),'k', LineWidth=2)
hold on
plot(x_est_store(2,:),'b--', LineWidth=2)
set(gca,'fontsize',16);
set(gca,'xticklabel',{[]})
ylabel('$x_2$', Interpreter='latex', fontsize=22)
grid on
ylim([-4 5.5])
legend('$x_{2,k}$', '$\hat x_{2,k|k}$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=18)

subplot(3,1,3)
semilogy(estimation_error_norm_squared,'b', LineWidth=2)
set(gca,'fontsize',16);
xlabel('$k$', Interpreter='latex', fontsize=22)
grid on
set(gca, 'YTick', 10.^(-11:2:10))
ylim([1e-5 1e5])
hold on
semilogy(J_cost,'r', LineWidth=2)
legend('${||e_{k|k}||}^{2}$', '${\rm{tr}} (P_{k|k})$', Location='best', box = 'off', NumColumns=2, Interpreter='latex', fontsize=18)

% print(gcf,'-dpng','Figures/Pendulum/png/Pendulum_NN_EKF')
% print(gcf,'-depsc','Figures/Pendulum/eps/Pendulum_NN_EKF')
% print(gcf,'-dpng','Figures/Pendulum/png/Pendulum_NN')
% print(gcf,'-depsc','Figures/Pendulum/eps/Pendulum_NN')

% training and validation losses
windowSize = 500;

smoothTrainLoss = movmean(trainLoss, windowSize);
smoothValLoss = movmean(valLoss, windowSize);

figure;
semilogy((1:1:numel(smoothTrainLoss))*validation_freq, smoothTrainLoss, 'r-', 'DisplayName', 'Training Loss', 'LineWidth',2);
hold on;
semilogy((1:1:numel(smoothValLoss))*validation_freq, smoothValLoss, 'b-', 'DisplayName', 'Validation Loss', 'LineWidth',2);
xlabel('Iteration');
ylabel('Loss');
axis tight
legend;
grid on;

% print(gcf,'-dpng','Figures/Pendulum/png/Pendulum_training_process')
% print(gcf,'-depsc','Figures/Pendulum/eps/Pendulum_training_process')