function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== Added section by MuammetDurmaz======================

for i = 1:m
   Error=theta(1,1)+theta(2,1)*X(i,2)-y(i,1);
   Ersqr2 = Error*Error/2; % square of error over 2
   J=J+Ersqr2/m;
   
end



% =========================================================================

end
