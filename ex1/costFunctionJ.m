function J = costFunctionJ(X,y,thetha)

% Linear regression cost function
% X is the m x n+1 design matrix 
% Thetha is the n+1 x 1 parameter matrix
% y is the m x 1 Training set
% Cost function is mean squared error between hypothesis & training set

m=size(X,1); % number of training examples

H = X*thetha; % H should be m x 1

SqErr = (H - y).^2;

J = sum(SqErr)/(2*m);