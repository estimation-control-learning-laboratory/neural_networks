function [K,P,J] = Compute_KPJ_EKF(A, C, Q, R, P)
%%% Computes the Kalman gain K, the prioir and the posterior estimation covariance P, and J%%%
P = A*P*A' + Q;             % Prior covariance
K = (P*C')/(C*P*C'+ R);     % Kalman gain
P = P - K*C*P;              % Posterior covariance
J = trace(P);
