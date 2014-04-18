function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); % size of training set
K = size(all_theta, 1); %number of labels

% You need to return the following variables correctly 
p = zeros(m, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X]; % X is now the design matrix of dimensions m x n+1

% The task of this function is to predict the classify each of m examples to 
% one of K classes for y. 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

rawpredictions = sigmoid(X*all_theta'); %m x K matrix
% for each of the m training examples, rawpredictions contains probability that y=i (i from 1 to K)
% we just need to take maximum of the probabilities for each training example to classify y


[maxval indexofmaxval] = max(rawpredictions,[],2);

% In this example, since the locations k=1:K also represent the classes, the 
% index of the maximum value will also represent the class k that we need to associate y with. 

p = indexofmaxval;

% =========================================================================


end
