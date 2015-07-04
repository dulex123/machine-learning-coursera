function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

width = size(z)(1,1);
height = size(z)(1,2);


for i=1:width,
	for j=1:height,
		g(i,j) += 1/(1+1/(e.^z(i,j)));
	end
end

% =============================================================

end
