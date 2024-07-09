clc; clear; close all
% Loading the training and the test data sets
addpath DataSets\VDP\
load('VDP_train_data_tf_0_1_ns_30000.mat')

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

lgraph = layerGraph(layers);

% Tran and Test Data
train_idx = size(x_data,1)*0.8;

train_inputs  = x_data(1:train_idx,:);
train_targets = y_data(1:train_idx,:);

val_inputs  = x_data(train_idx+1:end,:);
val_targets = y_data(train_idx+1:end,:);

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
% save('VDP_trained_network_v2.mat','net')

% Using a trained model
load VDP_trained_network.mat

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

%% Calculate mean squared error
mse = mean((predicted_outputs - test_targets).^2);
disp(['Validation MSE: ', num2str(mse)]);


