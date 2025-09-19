% safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
% logphi(z) = log(normcdf(z))
function lp = logphi(z)
  lp = zeros(size(z));                                         % allocate memory
  zmin = -6.2; zmax = -5.5;
  ok = z>zmax;                                % safe evaluation for large values
  bd = z<zmin;                                                 % use asymptotics
  ip = ~ok & ~bd;                             % interpolate between both of them
  lam = 1./(1+exp( 25*(1/2-(z(ip)-zmin)/(zmax-zmin)) ));       % interp. weights
  lp( ok) = log( (1+erf(z(ok)/sqrt(2)))/2 );
  % use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
  % lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
  % upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
  % the lower bound captures the asymptotics
  lp(~ok) = -log(pi)/2 -z(~ok).^2/2 -log( sqrt(z(~ok).^2/2+2)-z(~ok)/sqrt(2) );
  lp( ip) = (1-lam).*lp(ip) + lam.*log( (1+erf(z(ip)/sqrt(2)))/2 );