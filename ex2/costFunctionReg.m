function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
nplusone = length(theta); % n being number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% COMPUTING COST FUNCTION

% Cost Function #1: Need to compute Hypothesis function - h

thetaTransposeX = X*theta; % mx1 matrix
h = sigmoid(thetaTransposeX); % mx1 matrix with values between 0 and 1

% Cost Function #2: Compute actual regularized cost function per formula below

J = sum(-1.*y.*log(h) .- (1.-y).*log(1.-h))/m + (sum(theta.^2)-theta(1).^2)*lambda/(2*m);

% COMPUTING GRADIENT (DELTA) of Regularized Cost Function

grad = zeros(nplusone,1);

for k = 1:nplusone
	
	for l = 1:m
		grad(k) = grad(k)+ (h(l)-y(l))*X(l,k)*(1/m);	
	end
	
	if (k>1)
		grad(k) = grad(k) + (lambda/m)*theta(k);
	endif	
end




% =============================================================

end
