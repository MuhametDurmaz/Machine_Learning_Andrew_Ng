function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=size(X,2);
%------------This part is written by Muhammet Durmaz--------------

for iter = 1:num_iters
error=zeros(n,1)
   for i=1:m
   h_x=0   
       for t=1:n
           h_x=h_x+theta(t,1)*X(i,t)
       end
       for t=1:n
           error(t,1)=error(t,1)+(h_x-y(i,1))*X(i,t);    
       end
   end
   for t=1:n
   theta(t,1)= theta(t,1)-alpha/m*error(t,1);
   end
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
