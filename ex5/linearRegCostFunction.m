function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
nplusone = length(theta); % where n is number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Computing regularized cost function J
	% X will be m x n+1; y will be m x 1, theta will be n+1 + 1

hoftheta = X*theta;

unregcost = sum((hoftheta-y).^2) * (1/(2*m));

reg = (lambda/(2*m))*(sum(theta.^2)- theta(1)^2);

J = unregcost + reg;


% Computing gradient delta
	% delta will also be a n+1 x 1 vector (since theta:=theta +alpha*delta)

for j=1:nplusone

	for i=1:m
		grad(j) = grad(j) + (hoftheta(i)-y(i))*X(i,j)*(1/m);
	end
	
	if(j>1)
	
		grad(j) = grad(j) + (lambda/m)*theta(j);
	endif	
end



% =========================================================================

grad = grad(:);

end
