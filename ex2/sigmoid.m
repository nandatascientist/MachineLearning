function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

zdim = size(z);

% You need to return the following variables correctly 
g = zeros(zdim);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if (zdim(1)==1 && zdim(2)==1) 
   g = 1/(1+e^-z);
else
   g = 1 ./ (1 .+ exp(-1.*z));


% =============================================================

end
