function K = covSEard_GPD(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

%--------------2014-11-22--------------------------------------------------
% This is revised to include derivative  
% i indicates which coordiates to take derivative
% Only for the derivative moments
%--------------------------------------------------------------------------

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance
    
if nargin==4
   
   tempK3=sf2*exp(-sq_dists(diag(1./ell)*x',diag(1./ell)*z')./2);
   K=tempK3.*(1/ell(i)^2).*(1-sq_dists(x(:,i)'/ell(i),z(:,i)'/ell(i)));
  
end

a=1;