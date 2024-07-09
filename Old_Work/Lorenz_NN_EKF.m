clc; clear; close all
% Loading the training and the test data sets
addpath DataSets\Lorenz\
load('lorenz_train_data_tf_0_01_ns_20000.mat')
load('lorenz_test_data_tf_0_01_nt_1000.mat')

input_size  = size(x_train,2); 
output_size = size(y_train,2); 
hidden_size = 10;

% Define network architecture
layers = [
    featureInputLayer(input_size, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(hidden_size, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(hidden_size, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(hidden_size, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(hidden_size, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(output_size, 'Name', 'fc5')
    regressionLayer('Name', 'output')
];

layers_wo_output = layers(1:end-1);

lgraph = layerGraph(layers);
lgraph_wo_output = layerGraph(layers_wo_output);
dlnet = dlnetwork(lgraph_wo_output);

% Tran and Test Data
train_inputs  = x_train;
train_targets = y_train;

test_inputs  = x_test;
test_targets = y_test;

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Training the Model
% net = trainNetwork(train_inputs, train_targets, lgraph, options);
% save('Lorenz_trained_network.mat','net')

% Using a trained model
load Lorenz_trained_network.mat

%% Model Prediction Test
predicted_outputs = predict(net, test_inputs);

figure
set(gcf,'position',[200,100,1000,700])
subplot(3,1,1)
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,1),'b--', LineWidth=2)
ylabel('x')
grid on 
legend('$x$', '$\hat x$', Interpreter='latex', fontsize=10)
title('Predictions')

subplot(3,1,2)
plot(test_targets(:,2),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,2),'b--', LineWidth=2)
ylabel('y')
grid on
legend('$y$', '$\hat y$', Interpreter='latex', fontsize=10)

subplot(3,1,3)
plot(test_targets(:,3),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,3),'b--', LineWidth=2)
ylabel('z')
grid on
legend('$z$', '$\hat z$', Interpreter='latex', fontsize=10)

%% EKF Augmentation

% Initial states
x_dim = input_size;
num_samples = size(test_inputs,1);
x_est = test_inputs(1,:)';
P = eye(x_dim);

% Process and measurement noise
process_var = 0.1;
Q =  process_var*eye(x_dim);   
measurement_var = 0.001;
R = measurement_var;

% Initilization
x_est_store = zeros(x_dim, num_samples);
C = [1 0 0];
y_tot = zeros(size(C,1), num_samples);

% Estimation using EKF
for k = 1:num_samples
    inputData = dlarray(x_est, 'CB');
    [~, jacobian] = dlfeval(@computeJacobian, dlnet, inputData);
    A = jacobian;

    x_est  = predict(net, x_est')';   % Prior estimate

    [K, P, ~] = Compute_KPJ_EKF(A, C, Q, R, P);              % EKF: Prior & Posterior P and K caculation

    y = C*test_targets(k,:)' + sqrt(measurement_var)*randn;  % Measurement
    y_tot(k) = y;

    % x_est = x_est + K*(y - C*x_est);                         % Posterior estimate

    x_est_store(:, k) = x_est;
end


%% EKF Prediction Test
figure
set(gcf,'position',[200,100,800,300])
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(y_tot,'r--', LineWidth=2)
ylabel('x_1')
grid on 
legend('$x_1$', '$x_{1,m}$', Interpreter='latex', fontsize=10)
title('Measurements')


figure
set(gcf,'position',[200,100,1000,700])
subplot(3,1,1)
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(x_est_store(1,:),'b--', LineWidth=2)
ylabel('x')
grid on 
legend('$x$', '$\hat x$', Interpreter='latex', fontsize=10)
title('Predictions with EKF')

subplot(3,1,2)
plot(test_targets(:,2),'g', LineWidth=2)
hold on
plot(x_est_store(2,:),'b--', LineWidth=2)
ylabel('y')
grid on
legend('$y$', '$\hat y$', Interpreter='latex', fontsize=10)

subplot(3,1,3)
plot(test_targets(:,3),'g', LineWidth=2)
hold on
plot(x_est_store(3,:),'b--', LineWidth=2)
ylabel('z')
grid on
legend('$z$', '$\hat z$', Interpreter='latex', fontsize=10)

%% Calculate mean squared error
% mse = mean((predictions - trainTargetsVal).^2);
% disp(['Validation MSE: ', num2str(mse)]);



%% Define the function to compute gradients
function [output, jacobian] = computeJacobian(net, inputData)
    % Forward pass
    output = forward(net, inputData);

    % Initialize jacobian matrix
    numOutputs = size(output, 1);
    numInputs = size(inputData, 1);
    jacobian = zeros(numOutputs, numInputs);
    
    % Compute gradient for each output element w.r.t. inputs
    for i = 1:numOutputs
        gradOutput = zeros(numOutputs, 1, 'like', output);
        gradOutput(i) = 1;
        gradients = dlgradient(sum(output .* gradOutput), inputData, 'RetainData', true);
        jacobian(i, :) = extractdata(gradients);
    end
end



