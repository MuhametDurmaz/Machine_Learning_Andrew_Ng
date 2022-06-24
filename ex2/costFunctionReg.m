function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


% ====================== The code written by Muhammet Durmaz ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J = 0;
grad = zeros(size(theta));
n=length(theta(:,1));
J_theta=0;
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
    for t=2:1:n
        J_theta=J_theta+(theta(t,1)^2);
        grad(t,1)=grad(t,1)+lambda*theta(t,1)/m;
    end
J=J+J_theta*lambda/(2*m);






% =============================================================

end
