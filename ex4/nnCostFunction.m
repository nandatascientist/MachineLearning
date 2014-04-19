function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 3 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% ===================================================================================

% modify X to include the column of 1's
X = [ones(m,1) X] % x is our usual mxn+1 matrix now

nplusone = size(X,2);

% convert each instance of y into vector of length num_labels
yvect = zeros(m,num_labels) % each row represents a training example with num_label classes

% in our example m=5000 & num_labels=10.  y will be 5000x10 vector

for i=1:m
  
  ithlabel = y(i);
  yvect(i,ithlabel)=1;
  
end
% yvect is a m x num_labels matrix that we will use for backprop and training

% Cost is computed as function of hoftheta and yvect. hoftheta needs to be computed.

%=========== FORWARD & BACK PROPAGATION <START> =================

% we know that this is a 3 layer network. so we need a1,a2,a3 and z2,z3.
% hoftheta = a3, a3=g(z3) = g(theta2*a2)
% a2 = g(z2) = g(theta1*x)
% lets compute hoftheta for each example of i=1:m

% m=5000, n=400, firstlayersize=400, hiddenlayer=25,outputlayer=10
% we need z2 to be a 25x1 vector and z3 to be a 10x1 vector


hoftheta=zeros(m,num_labels);				%Hypothesis functino

DELTA1 = zeros(hidden_layer_size,input_layer_size+1);	%Incrementor for Theta1
DELTA2 = zeros(num_labels,hidden_layer_size+1);		%Incrementor for Theta1


for i=1:m

	%Theta1 is 25x401
	%Theta2 is 10x26

	z2 = Theta1*X(i,:)';   		% since x(i,)' is 401x1, z2 is 25x1
	a2 = sigmoid(z2);      		% a2 will be 25x1
	a2 = [1;a2];           		% a2 will be 26x1

	z3= Theta2*a2;         		% z3 will be 10x1
	a3 = sigmoid(z3);      		% a3 will be 10x1, where 10 is num_labels

	hoftheta(i,:)=a3';     		% hoftheta is the forward propagated matrix e.g. 5000x10
	
	%========< back propagation code >========================%
	
	del3 = a3-yvect(i,:)';		% del3 is a 10x1 vector
	
	%Theta2'*del3.*g'(z2)		% Theta2'*del3 is 26x1 vector
	one = ones(size(a2));		%precompute gprimez2 = a2 .*(1-a2)	
	gprimez2 = a2.*(one-a2);
	
	del2 = Theta2'*del3.*gprimez2;	%del2 is 26x1
	
	DELTA1 = DELTA1 + del2(2:hidden_layer_size+1)*X(i,:);  % <<<< 25x401 
	DELTA2 = DELTA2 + del3*a2';			       %10x1 times 1x26 >>10x26

	%========< back propagation code >========================%

end

D1 = (1/m).*DELTA1;
D2 = (1/m).*DELTA2;


%=========== FORWARD & BACK PROPAGATION <COMPLETE> =================

%=========== UNREGULARIZED COST <START> =====================

% initialize totalcost tcost

tcost=0;

for i=1:m

	iexamplecost = 0;
	% computed cost of the ith example across k labels	
	
	for k=1:num_labels

		iexamplecost = iexamplecost + yvect(i,k)*log(hoftheta(i,k)) + (1-yvect(i,k))*log(1-hoftheta(i,k));

	end


	tcost = tcost+iexamplecost;
end

J = (-1)*(1/m)*tcost;

%=========== UNREGULARIZED COST <COMPLETE> =====================


%=========== REGULARIZATION FOR COST <START> ==============================

% Theta1 is 25 x 401 and Theta2 is 10x26 

regterm = 0;

%use theta1 where l=1, s(l)=400, s(l+1)=25, dim 25x401

for i=2:input_layer_size+1

	for j=1:hidden_layer_size
		
		regterm = regterm + Theta1(j,i)^2;
		
	end

end

%use theta2 where l=2, s(l)=26, s(l+1)=10, dim 10x26

for i=2:hidden_layer_size+1

	for j=1:num_labels

		regterm = regterm + Theta2(j,i)^2;
	
	end

end


regterm = (lambda/(2*m))*regterm;

J  = J + regterm;

%=========== REGULARIZATION FOR COST <END> =====================

%=========== REGULARIZATION FOR GRAD <START> =====================

%Theta1 & D1 is 25x401
%Theta2 & D2 is 10x26

% i>25,j>401

tmpD1 = zeros(hidden_layer_size,input_layer_size+1);
tmpD2 = zeros(num_labels,hidden_layer_size+1);

for i=1:hidden_layer_size

	for j=2:input_layer_size+1
	

		tmpD1(i,j) = tmpD1(i,j) + (lambda/m)*Theta1(i,j);
	
	end

end

for i=1:num_labels

	for j=2:hidden_layer_size+1
	
		
		tmpD2(i,j) = tmpD2(i,j) + (lambda/m)*Theta2(i,j);
	end

end

%=========== REGULARIZATION FOR GRAD <END> =====================

Theta1_grad = D1 + tmpD1;
Theta2_grad = D2 + tmpD2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
