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


tempTheta = theta .^ 2;
tempTheta(1,1) = sqrt(tempTheta(1,1));
J = - 1/m * (log(sigmoid(X*theta))' * y + log(1-sigmoid(X*theta))' * (1-y)) + lambda/(2*m) * (sum(tempTheta) - tempTheta(1,1));


% First theta is computed differently by equation -> we do not regularize it
sum0=0;
for i=1:m,
	sum0 = sum0 + ( sigmoid(X(i,:)*theta) - y(i,1)) * X(i,1);
end
theta0 = 1/m * sum0;

% We need to loop trough all features
loopNum = rows(theta);

% And we need simultaneous updates so we use tempTheta
tempTheta = theta;
for i=2:loopNum,
	sum=0;
	for j=1:m,
		sum = sum + (sigmoid(X(j,:)*theta) - y(j,1) )* X(j,i);
	end
	tempTheta(i,1) = 1/m * sum + lambda/m*theta(i,1);  
end

% After the loop we update grad AND theta as it is requested
grad = theta = tempTheta;

% And at last we add theta0 
grad(1,1) = theta(1,1) = theta0;

% =============================================================

end
