function [K_1,K_2,K_3,KK_2] = covGP(loghyper, x, z, t)

% Squared Exponential covariance function with isotropic distance measure. The 
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix and sf2 is the signal
% variance. The hyperparameters are:
%
% loghyper = [ log(ell)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2007-06-25)
   ell = exp(loghyper(1));                           % characteristic length scale
   sf2 = exp(2*loghyper(2));                                     % signal variance

if nargin == 3
   K_1 = sf2*exp(-(gp_dist(x'/ell)).^2/2);
   tempK2=gp_dist(x'/ell,z'/ell);
   K_2 = sf2*exp(-tempK2.^2/2).*(-1/ell).*tempK2;
%   tempK22=gp_dist(z'/ell,x'/ell);
%   KK_2 = sf2*exp(-tempK2.^2/2).*(1/ell).*tempK2;
   KK_2=K_2';
   tempK3=gp_dist(z'/ell).^2;
   K_3 = sf2*exp(-tempK3./2).*1/ell^2.*(1-tempK3);
else
   sigma2_n=exp(2*loghyper(3));
   K_1 = sf2*exp(-(gp_dist(t'/ell)).^2/2);
   tempK21=gp_dist(t'/ell,z'/ell);
   tempK22=gp_dist(t'/ell,x'/ell);
   K_2 = [sf2*exp(-tempK21.^2/2).*(-1/ell).*tempK21,sf2*exp(-tempK22.^2/2)];
   KK_2 = K_2';
   tempK31=gp_dist(z'/ell).^2;
   tempK32=gp_dist(z'/ell,x'/ell);
   tempK33=sf2*exp(-tempK32.^2/2).*(1/ell).*tempK32;
   K_3 = [sf2*exp(-tempK31./2).*1/ell^2.*(1-tempK31),tempK33;...
          tempK33',sigma2_n*diag(ones(size(x,1),1))+sf2*exp(-(gp_dist(x'/ell)).^2/2)];
end
       
 

