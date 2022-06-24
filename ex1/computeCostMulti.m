function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
% You need to return the following variables correctly 
J = 0;

% ====================== This Part is Written by Muhammet Durmaz ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
h_x=zeros(m,1);
for i = 1:m
    for j= 1:n % function generate
   h_x(i,1)=h_x(i,1)+theta(j,1)*X(i,j);
    end
    Error=h_x(i,1)-y(i,1)
    Ersqr2 = Error*Error/2; % square of error over 2
   J=J+Ersqr2/m;
    
end




% =========================================================================

end
