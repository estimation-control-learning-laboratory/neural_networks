clc; clear; close all
load('lorenz_test_data_dt_0_01_T_10.mat')
load('x_est_store_w_EKF_file.mat')
load('x_est_store_wo_EKF_file.mat')
test_targets = y_test;
% Define the number of points for the tail
tailLength = 10;

% Define the figure
figure;
hold on;
grid on;

% Plot the static part
plot3(test_targets(:,1), test_targets(:,2), test_targets(:,3), 'k');
view(30,20)

% Initialize the moving parts
h1 = plot3(x_est_store_w_EKF(1,1), x_est_store_w_EKF(2,1), x_est_store_w_EKF(3,1), 'b--o');
h2 = plot3(x_est_store_wo_EKF(1,1), x_est_store_wo_EKF(2,1), x_est_store_wo_EKF(3,1), 'r--o');

legend('$x_{k}$', '$\hat x_{k|k}^{NN-EKF}$', '$\hat x_{k|k}^{NN}$', Location='best', box = 'off', NumColumns=3, Interpreter='latex', fontsize=12)


% Set axis limits
axis([min(test_targets(:,1)) max(test_targets(:,1)) min(test_targets(:,2)) max(test_targets(:,2)) min(test_targets(:,3)) max(test_targets(:,3))]);

% Prepare for GIF creation
filename = '3D_animation.gif';
frameDelay = 0.01;

for k = 1:length(test_targets)
    % Update the data for the tails and markers
    set(h1, 'XData', x_est_store_w_EKF(1, max(1,k-tailLength):k), ...
            'YData', x_est_store_w_EKF(2, max(1,k-tailLength):k), ...
            'ZData', x_est_store_w_EKF(3, max(1,k-tailLength):k));
        
    set(h2, 'XData', x_est_store_wo_EKF(1, max(1,k-tailLength):k), ...
            'YData', x_est_store_wo_EKF(2, max(1,k-tailLength):k), ...
            'ZData', x_est_store_wo_EKF(3, max(1,k-tailLength):k));

    % Draw the updated plot
    drawnow;

    % Capture the frame
    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);

    % Write to the GIF File
    if k == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', frameDelay);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', frameDelay);
    end
end
