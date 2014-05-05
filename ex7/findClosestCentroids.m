function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % number of desired centroids

% Set m
m = size(X,1); % number of training examples

% You need to return the following variables correctly.

idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% X is a mxn matrix with each example being represented in a row
% centroids is a Kxn matrix

% for each of training examples i.e. xi, we need to compute the distance between xi and mu1,...muK
% we will assign that mu that has lowest distance.

distanceVector = zeros(K,1);
distance = 0;

for i=1:m


	for j=1:K
	
		xi_minus_muj = X(i,:)- centroids(j,:); # this should be a n dim vector 
		distanceVector(j) = sum(xi_minus_muj.^2); 
	
	end

	# distanceVectorj should be populated with squared norm of distance between xi and all K centroids

	[distance ind] = min(distanceVector);
	idx(i) = ind;
end



% =============================================================

end

