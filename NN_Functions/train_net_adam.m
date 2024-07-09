function [dlnet,iteration,trainLoss,valLoss] = train_net_adam(dlnet, train_inputs, train_targets, val_inputs, val_targets, epochs, batch_size, learning_rate_init, validation_freq, shuffle)

% Convert data to dlarray
% dlTrainX = dlarray(train_inputs', 'CB');
% dlTrainY = dlarray(train_targets', 'CB');
dlValX = dlarray(val_inputs', 'CB');
dlValY = dlarray(val_targets', 'CB');

% Initialize training progress
iteration = 0;
trainLoss = [];
valLoss = [];

% Initialize the optimizer parameters
trailingAvg = [];
trailingAvgSq = [];
gradDecay = 0.9;
gradDecaySq = 0.999;

% Training loop
for epoch = 1:epochs
    % Shuffle data at the beginning of each epoch
    if shuffle == true
        idx = randperm(size(train_inputs, 1));
        dlTrainX = dlarray(train_inputs(idx, :)', 'CB');
        dlTrainY = dlarray(train_targets(idx, :)', 'CB');
    else
        dlTrainX = dlarray(train_inputs', 'CB');
        dlTrainY = dlarray(train_targets', 'CB');
    end

    % Mini-batch loop
    for i = 1:batch_size:size(train_inputs, 1)-batch_size+1
        iteration = iteration + 1;

        % Get mini-batch
        idxBatch = i:i+batch_size-1;
        dlXBatch = dlTrainX(:, idxBatch);
        dlYBatch = dlTrainY(:, idxBatch);

        % Evaluate the model gradients and loss using dlfeval and the modelGradients function
        [loss, gradients] = dlfeval(@modelGradients, dlnet, dlXBatch, dlYBatch);

        % Update the network parameters using the Adam optimizer
        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, trailingAvg, trailingAvgSq, iteration, learning_rate_init, gradDecay, gradDecaySq);

        % Store the training loss
        % trainLoss(end+1) = double(gather(loss)); % moved inside the "if"
        % condition so that it is saved at the validation frequency rate
      

        % Validate the network
        if mod(iteration, validation_freq) == 0
            dlYPredVal = predict(dlnet, dlValX);
            trainLoss(end+1) = double(gather(loss)); % it was moved to here
            valLoss(end+1) = mse(dlYPredVal, dlValY);
            fprintf('Epoch %d, Iteration %d, Training Loss: %.4f, Validation Loss: %.4f\n', epoch, iteration, trainLoss(end), valLoss(end));
        end
    end
end
end

% Custom function to compute model gradients and loss
function [loss, gradients] = modelGradients(dlnet, dlX, dlY)
dlYPred = forward(dlnet, dlX);
loss = mse(dlYPred, dlY);
gradients = dlgradient(loss, dlnet.Learnables);
end

% Function to compute the mean squared error loss
function loss = mse(dlYPred, dlY)
loss = mean((dlYPred - dlY).^2, 'all');
end
