function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


J = - 1/m * (log(sigmoid(X*theta))' * y + log(1-sigmoid(X*theta))' * (1-y));

sum0=0;
sum1=0;
sum2=0;

  for i=1:m,
  	sum0 = sum0 + ( sigmoid(X(i,:)*theta) - y(i,1)) * X(i,1);
  end

  for i=1:m,
    sum1 = sum1 + ( sigmoid(X(i,:)*theta) - y(i,1)) * X(i,2);
  end

  for i=1:m,
    sum2 = sum2 + ( sigmoid(X(i,:)*theta) - y(i,1)) * X(i,3);
  end

  

grad(1,1) = theta(1,1) = 1/m * sum0;
grad(2,1) = theta(2,1) = 1/m * sum1;
grad(3,1) = theta(3,1) = 1/m * sum2;

% =============================================================

end
