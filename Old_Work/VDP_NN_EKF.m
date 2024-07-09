clc; clear; close all
% Loading the training and the test data sets
addpath DataSets\VDP\
load('VDP_train_data_tf_0_1_ns_30000.mat')
load('VDP_test_data_tf_0_1_nt_500.mat')

input_size  = size(x_data,2); 
output_size = size(y_data,2); 
hidden_size = 10;

% Define network architecture
layers = [
    featureInputLayer(input_size, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(hidden_size, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(hidden_size, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(output_size, 'Name', 'fc3')
    regressionLayer('Name', 'output')
];
layers_wo_output = layers(1:end-1);

lgraph = layerGraph(layers);
lgraph_wo_output = layerGraph(layers_wo_output);

dlnet = dlnetwork(lgraph_wo_output);

% Tran and Test Data
train_idx = size(x_data,1)*0.8;

train_inputs  = x_data(1:train_idx,:);
train_targets = y_data(1:train_idx,:);

val_inputs  = x_data(train_idx+1:end,:);
val_targets = y_data(train_idx+1:end,:);

test_inputs  = x_test;
test_targets = y_test;

% Training options
validation_freq = 30;
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {val_inputs, val_targets}, ...
    'ValidationFrequency', validation_freq, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputFcn', @plotTrainingProgress);

% Training the Model
% net = trainNetwork(train_inputs, train_targets, lgraph, options);
% save('VDP_trained_network.mat','net')
% save('VDP_train_val_loss.mat', 'valLoss', 'trainLoss')

% Using a trained model
load VDP_trained_network.mat
load VDP_train_val_loss.mat

%% Model Prediction Test
predicted_outputs = predict(net, test_inputs);

figure
set(gcf,'position',[200,100,1000,700])
subplot(2,1,1)
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,1),'b--', LineWidth=2)
ylabel('x_1')
grid on 
legend('$x_1$', '$\hat x_1$', Interpreter='latex', fontsize=10)
title('Predictions')

subplot(2,1,2)
plot(test_targets(:,2),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,2),'b--', LineWidth=2)
ylabel('x_2')
grid on
legend('$x_2$', '$\hat x_2$', Interpreter='latex', fontsize=10)

%% EKF Augmentation

% Initial states
x_dim = input_size;
num_samples = size(test_inputs,1);
x_est = test_inputs(1,:)';
P = eye(x_dim);

% Process and measurement noise
process_var = 0.0001;
Q =  process_var*eye(x_dim);   
measurement_var = 0.001;
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
    [~, jacobian] = dlfeval(@computeJacobian, dlnet, inputData);
    A = jacobian;

    eigs(:,k) = eig(A); 
    if norm(eigs(1,k))<1 && norm(eigs(2,k))<1
        count = count + 1;
    end

    x_est  = predict(net, x_est')';   % Prior estimate

    [K, P, J_cost(k)] = Compute_KPJ_EKF(A, C, Q, R, P);              % EKF: Prior & Posterior P and K caculation

    y = C*test_targets(k,:)' + sqrt(measurement_var)*randn;  % Measurement
    y_tot(k) = y;

    x_est = x_est + K*(y - C*x_est);                         % Posterior estimate

    estimation_error(:, k) = abs(x_est - test_targets(k,:)');   % Estimation Error
    estimation_error_norm_squared(:, k) = norm(x_est - test_targets(k,:)')^2; % Estimation Error Norm Squared

    x_est_store(:, k) = x_est;
end


%% EKF Prediction Test
% Eigen Values
figure
plot(real(eigs(1,:)), imag(eigs(1,:)),'bx')
hold on
plot(real(eigs(2,:)), imag(eigs(2,:)),'rx')
viscircles([0, 0], 1);
xlabel('real')
ylabel('imag')
grid on
axis equal
disp(['Number of stable eigen values:', num2str(count)])

% Measurement
figure
set(gcf,'position',[200,100,800,300])
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(y_tot,'r--', LineWidth=2)
ylabel('$x_{1,m}$', Interpreter='latex', fontsize=18)
xlabel('$k$', Interpreter='latex', fontsize=18)
grid on 
legend('$x_1$', '$x_{1,m}$', Interpreter='latex', fontsize=12)

% print(gcf,'-dpng','Figures/VDP/png/VDP_Measurements')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_Measurements')

% Predictions Plot Set 01
% figure
% % sgtitle('Predictions with NN')
% % set(gcf,'position',[200,100,1100,600])
% set(gcf,'position',[200,100,800,700])
% subplot(4,1,1)
% plot(test_targets(:,1),'k', LineWidth=2)
% hold on
% plot(x_est_store(1,:),'b--', LineWidth=2)
% set(gca,'fontsize',14);
% set(gca,'xticklabel',{[]})
% ylabel('$x_1$', Interpreter='latex', fontsize=18)
% % xlabel('$k$', Interpreter='latex', fontsize=18)
% grid on 
% ylim([-2.5 3.5])
% legend('$x_1$', '$\hat x_1$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=12)
% 
% subplot(4,1,2)
% plot(test_targets(:,2),'k', LineWidth=2)
% hold on
% plot(x_est_store(2,:),'b--', LineWidth=2)
% set(gca,'fontsize',14);
% set(gca,'xticklabel',{[]})
% ylabel('$x_2$', Interpreter='latex', fontsize=18)
% % xlabel('$k$', Interpreter='latex', fontsize=18)
% grid on
% ylim([-3 4])
% legend('$x_2$', '$\hat x_2$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=12)
% 
% subplot(4,1,3)
% semilogy(estimation_error(1,:),'b', LineWidth=2)
% set(gca,'fontsize',14);
% set(gca,'xticklabel',{[]})
% ylabel('$e_1$', Interpreter='latex', fontsize=18)
% % xlabel('$k$', Interpreter='latex', fontsize=18)
% grid on
% set(gca, 'YTick', 10.^(-11:2:10))
% ylim([1e-4 1e1])
% 
% subplot(4,1,4)
% semilogy(estimation_error(2,:),'b', LineWidth=2)
% set(gca,'fontsize',14);
% ylabel('$e_2$', Interpreter='latex', fontsize=18)
% xlabel('$k$', Interpreter='latex', fontsize=18)
% grid on
% set(gca, 'YTick', 10.^(-11:2:10))
% ylim([1e-4 1e1])

% Predictions Plot Set 02
figure
set(gcf,'position',[200,100,800,700])
subplot(3,1,1)
plot(test_targets(:,1),'k', LineWidth=2)
hold on
plot(x_est_store(1,:),'b--', LineWidth=2)
set(gca,'fontsize',14);
set(gca,'xticklabel',{[]})
ylabel('$x_1$', Interpreter='latex', fontsize=18)
grid on 
ylim([-2.5 3.5])
legend('$x_{1,k}$', '$\hat x_{1,k|k}$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=12)

subplot(3,1,2)
plot(test_targets(:,2),'k', LineWidth=2)
hold on
plot(x_est_store(2,:),'b--', LineWidth=2)
set(gca,'fontsize',14);
set(gca,'xticklabel',{[]})
ylabel('$x_2$', Interpreter='latex', fontsize=18)
grid on
ylim([-3 4])
legend('$x_{2,k}$', '$\hat x_{2,k|k}$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=12)

subplot(3,1,3)
semilogy(estimation_error_norm_squared,'b', LineWidth=2)
set(gca,'fontsize',14);
set(gca,'xticklabel',{[]})
xlabel('$k$', Interpreter='latex', fontsize=18)
grid on
set(gca, 'YTick', 10.^(-11:2:10))
ylim([1e-5 1e2])
hold on
semilogy(J_cost,'r', LineWidth=2)
legend('${||e_{k|k}||}^{2}$', '${\rm{tr}} (P_{k|k})$', Location='northwest', box = 'off', NumColumns=2, Interpreter='latex', fontsize=12)

% print(gcf,'-dpng','Figures/VDP/png/VDP_NN_EKF')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_NN_EKF')
% print(gcf,'-dpng','Figures/VDP/png/VDP_NN')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_NN')

% training and validation losses
figure;
semilogy((1:1:numel(valLoss))*validation_freq, trainLoss, 'r-', 'DisplayName', 'Training Loss', 'LineWidth',2);
hold on;
semilogy((1:1:numel(valLoss))*validation_freq, valLoss, 'b-', 'DisplayName', 'Validation Loss', 'LineWidth',2);
xlabel('Iteration');
ylabel('Loss');
axis tight
legend;
grid on;

% print(gcf,'-dpng','Figures/VDP/png/VDP_training_process')
% print(gcf,'-depsc','Figures/VDP/eps/VDP_training_process')
%% Calculate mean squared error
mse = mean((predicted_outputs - test_targets).^2);
disp(['Validation MSE: ', num2str(mse)]);

%% Define the function to compute gradients
% Jacobian Calculation
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

% Storing the loss data
function stop = plotTrainingProgress(info)
    stop = false;
    persistent trainLoss valLoss
    if info.State == "start"
        trainLoss = [];
        valLoss = [];
    elseif info.State == "iteration"
        if ~isempty(info.ValidationLoss)
            valLoss(end+1) = info.ValidationLoss;
            trainLoss(end+1) = info.TrainingLoss;
        end
    elseif info.State == "done"
        assignin('base', 'trainLoss', trainLoss);
        assignin('base', 'valLoss', valLoss);
    end
end



