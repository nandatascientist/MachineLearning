function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% X is a nm x nfeatures matrix
% Theta is a nm x nfeatures paramter matrix
            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% X is a nmovies x nfeatures matrix
% Theta is a nusers x nfeatures paramter matrix
% X * thetha' > nmovies x nusers

thetatransposex = X * Theta'; # Compute predicted value
diff= thetatransposex(R==1)-Y(R==1); # take only values that have ratings and get difference in predictions Vs labels
J =sum(diff.^2)*0.5; # compute cost


% Computing regularized cost
reg1 =0;
reg2 =0;

for j=1:num_users

	reg1 = reg1 + sum(Theta(j,:).^2);

end

for i=1:num_movies

	reg2 = reg2 + sum(X(i,:).^2);

end

J = J+0.5*lambda*(reg1+reg2);

newdiff = thetatransposex - Y; % num_mov x num_users
            
% computing X gradient
% diff is num_movies x num_users while Theta is num_users x num_features

for i=1:num_movies
	
	for n=1:num_features
	
		temp=0;
		
		for j=1:num_users
			
			if(R(i,j)==1)
			
				temp = temp + (thetatransposex(i,j)-Y(i,j))*Theta(j,n);
			end	

		end
		
		X_grad(i,n) = temp + lambda*X(i,n);
	end	

end

for j = 1:num_users

	for n=1:num_features
	
		temp=0;
		
		for i=1:num_movies
		
			if(R(i,j)==1)
		
				temp = temp + (thetatransposex(i,j)-Y(i,j))*X(i,n);
			end
		end
	
		Theta_grad(j,n)=temp + lambda*Theta(j,n);
	end


end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
