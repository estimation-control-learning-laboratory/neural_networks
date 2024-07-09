function [output, jacobian] = computeJacobian(dlnet, inputData)
%%% Calculates the Jacobian of a Neural Network %%%
%%% Input data must be of dlarray(test_inputs', 'CB') type %%%
%%% Network must be of type lgraph = layerGraph(layers) -> dlnet = dlnetwork(lgraph) %%%
% Forward pass
output = forward(dlnet, inputData);

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

% To note that we can do it analytically as well (refer to Jacobian_Validation):

% Analytical Jacobian Calculation for the Pendulum example: 2input, 10hidden, 2output

% function [output, jacobian] = computeJacobian_Analytical(dlnet, inputData)
%     W1 = extractdata(dlnet.Learnables.Value{1});
%     b1 = extractdata(dlnet.Learnables.Value{2});
%     W2 = extractdata(dlnet.Learnables.Value{3});
%     b2 = extractdata(dlnet.Learnables.Value{4});
% 
%     z1 = W1 * inputData + b1;
%     h = max(0, z1); 
%     output = W2 * h + b2;
% 
%     relu_prime = @(z) double(z > 0);
%     D_relu = diag(relu_prime(z1));
%     jacobian = W2 * D_relu * W1;
% end