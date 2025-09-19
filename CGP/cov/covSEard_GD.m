function K = covSEard_GD(hyp, x, z, i, j)

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

%--------------2014-10-28--------------------------------------------------
% This is revised to include derivative  
% i=1 indicates K^{10}
% i=2 indicates K^{11}
% j indicates which input is going to take derivative 
%--------------------------------------------------------------------------

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = 1; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dists(diag(1./ell)*x');
  else                                                   % cross covariances Kxz
    K = sq_dists(diag(1./ell)*x',diag(1./ell)*z');
  end
end

K = sf2*exp(-K/2);                                                  % covariance

if nargin>3                                                        % derivatives
  if i==1                                                       % K^{10}
      if xeqz
        K = K.*gp_dist(x(:,j)'/ell(j)).*(-1/ell(j));
      else
        K = K.*gp_dist(x(:,j)'/ell(j),z(:,j)'/ell(j)).*(-1/ell(j));
      end
  elseif i==2                                                      % K^{11}
      if xeqz
        K = K.*(1/ell(j)^2).*(1-sq_dists(x(:,j)'/ell(j)));
      else
        K = K.*(1/ell(j)^2).*(1-sq_dists(x(:,j)'/ell(j),z(:,j)'/ell(j)));
      end
  end
end