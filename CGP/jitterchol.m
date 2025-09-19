function chol_A=jitterchol(A)

%---------CopyRight @ Xiaojing Wang

%---------2014-11-04---------------
% Do Choleskey decomposition

%----------------------------------

jitter=1e-06;

[PX, err]= chol(A, 'lower');

if err>0;
   warning('Jitter Added'); %#ok<WNTAG>
   chol_A=chol(A+jitter*eye(size(A,1)), 'lower');
else
   chol_A=chol(A, 'lower');
end