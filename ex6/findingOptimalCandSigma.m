function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

lowestC = 0;
lowestsigma =0;

lowestError = 1000000000000;

cval = [0.01 0.03 0.1 0.3 1 3 10 30]
sigmaval = [0.01 0.03 0.1 0.3 1 3 10 30]

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% Train a SVM classifier with the training set and initial values of C,sigma
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict(model,Xval);
originalError = mean(double(predictions ~= yval));

lowestError = originalError; 

for i=1:8

	for j=1:8

		% Train a SVM classifier with the training set and initial values of C,sigma
		modelfit = svmTrain(X, y, cval(i), @(x1, x2) gaussianKernel(x1, x2, sigmaval(j))); 
		predictionscv = svmPredict(modelfit,Xval);
		err = mean(double(predictionscv ~= yval));
		
		if(err<lowestError)
			
			lowestError = err;
			lowestC = cval(i);
			lowestsigma = sigmaval(j);
		
		endif
	
	end

end


if (lowestError<originalError)

	C = lowestC;
	sigma = lowestsigma;

endif


% =========================================================================

end
