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
n=length(theta(:,1));
% ============= The code Written by Muhammet Durmaz ===============
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
    for dat=1:1:m % dat is the index of data
      h_theta=0;%theta*X function
        for thet=1:1:n
        h_theta=h_theta + theta(thet,1)* X(dat,thet);
        end
      h_thetaX=sigmoid(h_theta);%h_theta function
      %The value of h_theat represents the probability of output is 1. 
      error= -y(dat)*log(h_thetaX)-log(1-h_thetaX)+y(dat)*log(1-h_thetaX);
      ek=(h_thetaX-y(dat))*transpose(X(dat,:));
      grad=grad+ek/m;
      J=J+error/m;
    
    end







% =============================================================

end
