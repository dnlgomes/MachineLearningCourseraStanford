function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
h = X * theta;

sqr_error = (h - y) .^2;

sum_sqr_error = sum(sqr_error);


% You need to return the following variables correctly 
J = (1 / (2 * m)) * sum_sqr_error;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
