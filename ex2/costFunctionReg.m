function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
n = size(theta);

for i = 1:m
  J += -y(i) * log(h(i)) - (1 - y(i)) * log(1 - h(i));
endfor
J /= m;

for j = 2:n
  J += (lambda / (2 * m)) * theta(j) ^ 2;
endfor

for i = 1:m
  grad(1) += (1 / m) * (h(i) - y(i)) * X(i, 1); 
endfor

for j = 2:n
  for i = 1:m
    grad(j) += (h(i) - y(i)) * X(i, j) + (lambda / m) * theta(j);
  endfor
  grad(j) /= m;
endfor

% =============================================================

end
