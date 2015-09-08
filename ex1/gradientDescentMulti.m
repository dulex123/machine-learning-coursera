function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
loopNum = length(theta);
tempTheta = theta;

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    

    for i=1:loopNum,
        sum = 0;
        for j=1:m,
            sum = sum + ( X(j,:) * theta - y(j,1)) * X(j,i);
        end
        tempTheta(i,1) = theta(i,1) - alpha * 1/m * sum;
    end
    theta = tempTheta;


    % non-vectorized solution - works in matlab
    % for j=1:size(X,2)
    %     sum=0;
    %     for i=1:m,

    %         hipo = 0;
    %         for hj=1:size(X,2)
    %             hipo = hipo + theta(hj,1)*X(i,hj);
    %         end
    %         sum = sum + (hipo - y(i,1))*X(i,j);
    %         %sum = sum + (theta(1,1)*X(i,1)+theta(2,1)*X(i,2)+theta(3,1)*X(i,3) - y(i,1))*X(i,j);
    %     end;
    %     temptempTheta = theta';
    %     tempTheta(1,j)=temptempTheta(1,j)-alpha*1/m* sum;
    % end
    % theta = tempTheta';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
