function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values


for iter = 1:num_iters

    % ====================== This Part added by Muhammet Durmaz ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    Error=zeros(2,1);
    m=length(y);
    for i=1:m
 Error(1,1)=Error(1,1)+(theta(1,1)+theta(2,1)*X(i,2)-y(i,1));
 Error(2,1)=Error(2,1)+(theta(1,1)+theta(2,1)*X(i,2)-y(i,1))*X(i,2);
 
    end
theta(1,1)= theta(1,1)-alpha/m*Error(1,1);
theta(2,1)= theta(2,1)-alpha/m*Error(2,1);
  % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
