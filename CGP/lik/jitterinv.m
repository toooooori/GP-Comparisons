function inv_A=jitterinv(A)

jitter=1e-06;

[PX, err]= chol(A);

if err>0;
   warning('Jitter Added'); %#ok<WNTAG>
   inv_A=inv(A+jitter*eye(size(A,1)));
else
   inv_A=inv(A);
end