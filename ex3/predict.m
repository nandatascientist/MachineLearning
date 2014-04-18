function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Theta1 should be 26x401 vector if the only hidden layer has 25 units and X has 400 features
% Theta2 should be 10x26 vector if the output layer has 10 labels

% Dimensions
m = size(X, 1); % we assume X is a mx1 vector
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

X = [ones(m,1) X]; % X is now our usual design matrix of dimensions m x n+1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

z2 = X*Theta1';% 5000 x 26
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

z3 = a2*Theta2'; % 5000 x 10
a3 = sigmoid(z3);

[maxval indexmaxval] = max(a3,[],2);

p = indexmaxval;





% =========================================================================


end
