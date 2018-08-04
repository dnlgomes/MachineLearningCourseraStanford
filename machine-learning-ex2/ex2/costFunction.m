function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
z = X * theta;
h = sigmoid(z);

each = -y .* log(h) - (1 - y) .* log(1 - h);

% You need to return the following variables correctly 
J = (1 / m) * sum(each);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

theta0_d = (1 / m) * sum((h - y) .* X(:, 1));
theta1_d = (1 / m) * sum((h - y) .* X(:, 2));
theta2_d = (1 / m) * sum((h - y) .* X(:, 3));
grad = [theta0_d; theta1_d; theta2_d];







% =============================================================

end
