clc; clear; close all
% Loading the training and the test data sets
addpath DataSets\Pendulum\
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
    regressionLayer('Name', 'output')
];

lgraph = layerGraph(layers);

% Train, Validation, and Test Data
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
    'ValidationData', {val_inputs, val_targets}, ... % Provide validation data
    'ValidationFrequency', 30, ... % Set validation frequency
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'OutputFcn', @plotTrainingProgress);

% Training the Model
net = trainNetwork(train_inputs, train_targets, lgraph, options);
save('Pendulum_trained_network_v2.mat','net')

% Using a trained model
% load Pendulum_trained_network_v2.mat

%% Model Prediction Test
predicted_outputs = predict(net, test_inputs);

figure
set(gcf,'position',[200,100,1000,700])
subplot(2,1,1)
plot(test_targets(:,1),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,1),'b--', LineWidth=2)
ylabel('Angle')
grid on 
legend('$\theta$', '$\hat \theta$', Interpreter='latex', fontsize=10)
title('Predictions')

subplot(2,1,2)
plot(test_targets(:,2),'g', LineWidth=2)
hold on
plot(predicted_outputs(:,2),'b--', LineWidth=2)
ylabel('Angular Velocity')
grid on
legend('$\dot \theta$', '$\hat{\dot{\theta}}$', Interpreter='latex', fontsize=10)

%% Calculate mean squared error
% mse = mean((predicted_outputs - test_targets).^2);
% disp(['Validation MSE: ', num2str(mse)]);

% Plot the training and validation losses
figure;
semilogy((1:1:numel(valLoss))*30, trainLoss, '-', 'DisplayName', 'Training Loss', 'LineWidth',2);
hold on;
semilogy((1:1:numel(valLoss))*30, valLoss, '-', 'DisplayName', 'Validation Loss', 'LineWidth',2);
xlabel('Iteration');
ylabel('Loss');
title('Training and Validation Loss');
legend;
grid on;

function stop = plotTrainingProgress(info)
    stop = false;  % Set to true to stop training early

    % Store training progress in persistent variables
    persistent trainLoss valLoss
    if info.State == "start"
        % Initialize variables at the start of training
        trainLoss = [];
        valLoss = [];
    elseif info.State == "iteration"
        % Append the current training loss
        
        % Append the current validation loss if it's available
        if ~isempty(info.ValidationLoss)
            valLoss(end+1) = info.ValidationLoss;
            trainLoss(end+1) = info.TrainingLoss;
        end
    elseif info.State == "done"
        % Save the training and validation loss to the workspace
        assignin('base', 'trainLoss', trainLoss);
        assignin('base', 'valLoss', valLoss);
    end
end








