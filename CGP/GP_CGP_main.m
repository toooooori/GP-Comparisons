%------------CopyRight @ Xiaojing Wang-----------------------

GPalg_prgp;

t2=sort(t2);
m =length(t2);
tstar=tstar_g;

%------------simulation GP data-------------------------------

cov_X=feval(covfunc{:}, loghyper, t1);
cov_ZT=covSEisoS([log(ell);log(sigma_f)],t1);                    % prior covariance matrix of Z(T)



[K1,K2,K3,KK2]=covGP([log(ell);log(sigma_f)],t1,t2);
L_covX=jitterchol(cov_X);
alpha_covX=L_covX'\(L_covX\(x-mu));
L_K3=jitterchol(K3);

mean_dX=(L_K3'\(L_K3\KK2))';                                                % A matrix
cov_dX=cov_X-mean_dX*KK2;                                                   % B matrix                                                                               


L_dX=jitterchol(cov_dX);


[Ks1,Ks2,Ks3,KKs2]=covGP([log(ell);log(sigma_f)],tstar,t2);                 % prior covariance matrix of Z(T*), Z'(S)
cov_Zxt=covSEisoS([log(ell);log(sigma_f)],t1,tstar);                         % prior covariance matrix of Z(T*), Z(T)
cov_Zts=covSEisoS([log(ell);log(sigma_f)],tstar);                            % prior covariance matrix of Z(T*)


Sigma_11=[cov_X, cov_Zxt; cov_Zxt', cov_Zts] ;
Sigma_12=[K2; Ks2];
Sigma_21=[KK2, KKs2];
Sigma_22=K3;

v_K3=L_K3\Sigma_21;


Lambda=Sigma_11-v_K3'*v_K3;

Lambda_11=Lambda(1:n,1:n);
Lambda_12=Lambda(1:n, (n+1):(n+nstar));
Lambda_21=Lambda((n+1):(n+nstar), 1:n);
Lambda_22=Lambda((n+1):(n+nstar), (n+1):(n+nstar));

L_Lb=jitterchol(Lambda_11);
v_Lb=L_Lb\Lambda_12;


Lambda_sig=Lambda_22-v_Lb'*v_Lb;



%----------------Initial Conditions for Gibbs Sampling---------------------
M=100000;
Z=zeros(nstar,M);                                                            % indicate Z(s)
Zd=zeros(m,M);                                                               % indicate Z'(s)
Zd_p=zeros(m,M);                                                             % indicate Z'^{+}(s) 
jitter=10e-7;


%----------------------Gibbs Sampling Procedure----------------------------
for k=2:M
    

mu_m=zeros(m,1);
nu_m=zeros(m,1);

theta_m=zeros(m,1);
delta_m=zeros(m,1);

%--------------Gibbs sampling procedure-----------------------------------
for i=1:m

%------------obtain the conditional proposal distribution------------------


%------------unconstraint posterior proposal distribution------------------
    
    temp_s=K3(i,:);
    temp_s(i)=[];
    temp_S=K3;
    temp_S(i,:)=[];
    temp_S(:,i)=[];
    
%------------unconstraint prior proposal distribution----------------------
    
    temp_a=mean_dX(:,i);
    if i==1
       temp_A=mean_dX(:, (i+1:m));
    elseif i==m
       temp_A=mean_dX(:,(1:i-1));
    else
       temp_A=[mean_dX(:,(1:i-1)), mean_dX(:, (i+1:m))];
    end


    
    
    if i>1 && i<m
        temp_zp=[Zd((1:i-1),k);Zd((i+1:m),k-1)];
        temp_zpp=[Zd_p((1:i-1),k);Zd_p((i+1:m),k-1)];
        temp_zpp0=[Zd_p((1:i-1),k);0;Zd_p((i+1:m),k-1)];
    elseif i==1
        temp_zp=Zd(2:m,k-1);
        temp_zpp=Zd_p(2:m,k-1);
        temp_zpp0=[0;Zd_p(2:m,k-1)];
    else
        temp_zp=Zd(1:m-1,k);
        temp_zpp=Zd_p(1:m-1,k);
        temp_zpp0=[Zd_p(1:m-1,k);0];
    end
    
%--------------normal distribution for prior---------------------  

    tempS_chol=jitterchol(temp_S);
     
    v_tempS=tempS_chol\(temp_s');
    
    alpha_tempS=tempS_chol'\(tempS_chol\temp_zp);
    
    mu_m(i)=temp_s*alpha_tempS;
    nu_m(i)=K3(i,i)-v_tempS'*v_tempS;
    nu_sqrt=sqrt(nu_m(i));
    
%----------unconstraint posterior normal distribution------------    
    v_dX=L_dX\temp_a;  
       
    temp_th1=(x-mu)-temp_A*temp_zpp;
    
    alpha_dX=L_dX'\(L_dX\temp_th1);

    temp_th=temp_a'*alpha_dX+mu_m(i)*nu_m(i)^(-1);
    temp_dt1=nu_m(i)^(-1)+v_dX'*v_dX;
    temp_dt=1/temp_dt1;
    theta_m(i)=temp_dt*temp_th;
    delta_m(i)=sqrt(temp_dt);

%---------------Gibbs Sampling Procedure One--------------------

%---------------Binomial Selection Algorithm-------------------
  
    eps=0;    
    temp_ks=normcdf(eps,mu_m(i),nu_sqrt);

    temp_qx=1-normcdf(eps,theta_m(i),delta_m(i));    
    temp_const=temp_qx*sqrt(delta_m(i)/nu_sqrt)*exp(-mu_m(i)^2/(2*nu_m(i))+theta_m(i)^2/(2*temp_dt));

    q_const=temp_const/(temp_ks+temp_const);
    
     if temp_const==Inf   
       q_const=1;
    end
    
       
    if isnan(temp_const)
       warning('NaN value')
       break;
    end
    
    if temp_ks==0 && temp_const==0
       temp_ks=1e-6;
        q_const=temp_const/(temp_ks+temp_const);
    end
   
    biv_num=binornd(1,q_const);
    
    if biv_num==1
       u=temp_ks*rand(1);
       Zd(i,k)=norminv(u,mu_m(i),nu_sqrt);
       Zd_p(i,k)=0;
    else
       u=1-(temp_qx-temp_qx*rand(1));
       Zd(i,k)=norminv(u,theta_m(i),delta_m(i));
       Zd_p(i,k)=Zd(i,k);
    end
  
   
end

mu_lam=mu*ones(n+nstar,1);
alpha_K3=L_K3'\(L_K3\Zd_p(:,k));

MU=mu_lam+Sigma_12*alpha_K3;
mu_z=MU((n+1):(n+nstar));
mu_x=MU(1:n);

%---------------Gibbs Sampling Procedure Two--------------------
%---------------Multivariate Normal Distribution----------------

Z_mu=mu_z+Lambda_21*(L_Lb'\(L_Lb\(x-mu_x)));
Z_sigma=Lambda_sig;
Z(:,k)=Z_mu+chol(Z_sigma+jitter*eye(size(Z_sigma,1)))'*randn(nstar,1);

end
 
Zval=Z(:,50001:100000);  
Z_qt=quantile(Zval', [0.025,0.975]);  
Z_qt1=Z_qt(1,:);
Z_qt2=Z_qt(2,:);

Zd_pval=Zd_p(:,5001:10000);
Zd_qt=quantile(Zd_pval', [0.025,0.975]);  
Zd_qt1=Zd_qt(1,:);
Zd_qt2=Zd_qt(2,:);


if fm==1
   xstar=4./(1+exp(-tstar./2+4));
elseif fm==2
    xstar=sin(tstar)+tstar;
elseif fm==3
      xstar=zeros(nstar,1);
      for i=1:nstar
          if tstar(i)<-5
             xstar(i)=tstar(i)+5;
          elseif tstar(i)>5
             xstar(i)=tstar(i)-5;
          else
             xstar(i)=0;
          end
     end
end


disp(mean((median(Zval,2) - xstar).^2))
figure;
plot(tstar,median(Zval,2),'.b','MarkerSize', 17);
hold on;
plot(tstar,xstar,'.r','MarkerSize', 17);
plot(tstar,Z_qt1,'-.b','LineWidth', 1, 'MarkerSize', 17);
plot(tstar,Z_qt2,'-.b','LineWidth', 1, 'MarkerSize', 17);

plot(t1,x,'+k','MarkerSize', 13)
ay=axis;

plot(t2,ay(3),'.g','MarkerSize', 17) 

    set(h,'Interpreter','none')




