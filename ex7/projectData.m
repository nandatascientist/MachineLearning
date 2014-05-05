function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
m = size(X,1); % m is number of training examples
n = size(X,2); % n is number of features

% so X is a mxn matrix

Z = zeros(m, K); % Z will be a mxK matrix


% K needs to be lesser than n for "dimensionality reduction" to take place

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%


Ureduce = U(:,1:K); % this matrix has the first K eigenvectors from PCA
% Ureduce should be nxK

%  We compute Ureduce'*X'  => [Kxn] * [nxm] > K*m' and transpose this

calc = Ureduce'*X';
Z = calc';

% =============================================================


end
