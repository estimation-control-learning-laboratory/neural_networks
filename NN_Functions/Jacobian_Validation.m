clc; clear; close all
% rng('default'); 

layers = [
    featureInputLayer(2, 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(2, 'Name', 'fc1')
    sigmoidLayer('Name', 'sigmoid1')
    fullyConnectedLayer(2, 'Name', 'fc2')
];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

x = randn(2,1);
inputData = dlarray(x, 'CB');
[~, jacobian_computed] = dlfeval(@computeJacobian, net, inputData);

W1 = extractdata(net.Learnables.Value{1});
b1 = extractdata(net.Learnables.Value{2});
W2 = extractdata(net.Learnables.Value{3});
b2 = extractdata(net.Learnables.Value{4});

z1 = W1 * x + b1;
h = 1 ./ (1 + exp(-z1));
y = W2 * h + b2;

sigmoid_prime = @(z) (1 ./ (1 + exp(-z))) .* (1 - (1 ./ (1 + exp(-z))));
D_sigma = diag(sigmoid_prime(z1));
J_analytical = W2 * D_sigma * W1;

disp('Analytical Jacobian:');
disp(J_analytical);
disp('Computed Jacobian:');
disp(jacobian_computed);

if norm(J_analytical - jacobian_computed) < 1e-6
    disp("The Jacobians match!");
else
    disp("The Jacobians don't match!");
end


%% Functions
function [output, jacobian_computed] = computeJacobian(net, inputData)
    output = predict(net, inputData);

    numOutputs = numel(output);
    numInputs = numel(inputData);
    jacobian_computed = zeros(numOutputs, numInputs);

    for i = 1:numOutputs
        gradOutput = zeros(size(output), 'like', output);
        gradOutput(i) = 1;

        gradients = dlgradient(sum(output .* gradOutput), inputData, 'RetainData', true);

        jacobian_computed(i, :) = extractdata(gradients);
    end
end


