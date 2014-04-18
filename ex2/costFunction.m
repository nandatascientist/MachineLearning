function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
nplusone = length(theta); % where n is the number of features

thetatransposeX = zeros(m,1);
hoftheta = zeros(m,1);

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

% COMPUTING COST

% Compute hypothesis function hoftheta

thetatransposeX = X*theta;
hoftheta = sigmoid(thetatransposeX);

%disp(hoftheta);

% Computing Cost function: J = -y*log(h) - (1-y)*log(1-h)

J = sum(-1.*y.*log(hoftheta) .- (1.-y).*log(1.-hoftheta))/m;

% COMPUTING GRADIENT

grad = zeros(nplusone,1);

for k = 1:nplusone
	
	for l = 1:m
		grad(k) = grad(k)+ (hoftheta(l)-y(l))*X(l,k)*(1/m);	
	end
	
end


% =============================================================

end
