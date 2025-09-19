function [mse, nlpd, coverage, width, elapsedTime] = CGP(t1_in, x_in, tstar_in, y_true)
% CGP — Constrained GP (IP-like interface)
% Usage: [mse, nlpd, coverage, width, elapsedTime] = CGP(t1, x, tstar, y_true)
% 把你提供的两段脚本（自适应选点 + 大M Gibbs）完全整合为一次调用；
% 保留所有数值细节，只在末尾按潜在函数空间返回评估指标。
%
% 依赖：GPML（gp/minimize/infExact/meanConst/covSEisoS/covNoise/covSum/likGauss）
%      以及 covGP, gp_dist, jitterchol 等工具函数需在 path 上。
%
% 注意：为保证与多次运行实验一致，函数内部不重置随机种子；如需可重复，
%       请在函数外部设置 rand/randn 的 seed。

% ----------- I/O -----------
t1    = t1_in(:);
x     = x_in(:);
tstar = tstar_in(:);
n     = numel(t1);
nstar = numel(tstar);

tAll = tic;

% ============================================================
% 1) 超参学习（与你脚本一致：covSum{SEisoS + Noise}）
% ============================================================
meanfunc = {@meanConst}; hyp.mean = 0;
covfunc  = {'covSum',{'covSEisoS', 'covNoise'}};  hyp.cov = [0, 0, 0];
likfunc  = {@likGauss};  hyp.lik = -Inf;

[hyp_sol, ~, ~] = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, [], t1, x);

loghyper = hyp_sol.cov;
ell      = exp(hyp_sol.cov(1));
sigma_f  = exp(hyp_sol.cov(2));
mu       = hyp_sol.mean;

cov_X  = feval(covfunc{:}, loghyper, t1);
% 与脚本一致地计算但未直接使用：
cov_ZT = covSEiso([log(ell);log(sigma_f)],t1);

% ============================================================
% 2) 自适应选点阶段（与你第二段脚本流程一致）
% ============================================================
set_num=0;
[Ks1,Ks2,Ks3,KKs2]=covGP([log(ell);log(sigma_f)],t1,tstar);

L_covX=jitterchol(cov_X);
alpha_covX=L_covX'\(L_covX\(x-mu));
mu_dS=KKs2*alpha_covX;
v_covX=L_covX\Ks2;
cov_dS=Ks3-v_covX'*v_covX;

prob_y=normcdf(zeros(nstar-set_num,1),mu_dS,diag(sqrt(cov_dS)));

% figure; plot(tstar, prob_y, '.', 'MarkerSize',17) % 原脚本会画图，这里省略

th_hd=1e-20;
t2=[];
tstar_g=tstar;

while max(prob_y)>th_hd

set_num=set_num+1;    

temp=find(prob_y==max(prob_y));
lg=length(temp);
if lg~=1
   j=1;
   temp_idx=0;
   while j<lg
   a_mat=temp(j);
   for i=j:lg
       if temp(i)~=(a_mat+i-j) 
          temp_idx=i-j;      
          break;
       end
   end   
      if temp_idx==0
         j=i;
      else
         temp_idx=i-1;
         break;
      end
   end
   if temp_idx~=0
      if mod(temp_idx,2)==0
         temp_va=tstar(temp(temp_idx/2));
      else
         temp_va=tstar(temp((temp_idx+1)/2));
      end
   else
      if mod(lg,2)==0
         temp_va=tstar(temp(lg/2));
      else
         temp_va=tstar(temp((lg+1)/2));
      end
   end
else
   temp_va=tstar(temp);
end


    t2=[t2;temp_va];

[K1,K2,K3,KK2]=covGP([log(ell);log(sigma_f)],t1,t2);

tstar=setdiff(tstar,t2);
lg_star=length(tstar);

[Ks1,Ks2,Ks3,KKs2]=covGP([log(ell);log(sigma_f)],t1,tstar);
tempK2=gp_dist(t2'/ell,tstar'/ell);
K_3 = sigma_f^2*exp(-tempK2.^2/2).*1/ell^2.*(1-tempK2.^2);

Sigma_11=[cov_X, Ks2; KKs2, Ks3] ;
Sigma_12=[K2; K_3'];
Sigma_21=[KK2, K_3];
Sigma_22=K3;
          
L_K3=jitterchol(K3);
v_K3=L_K3\Sigma_21;

    mu_lam=[mu*ones(n,1);zeros(lg_star,1)];
Lambda=Sigma_11-v_K3'*v_K3;

Lambda_11=Lambda(1:n,1:n);
Lambda_12=Lambda(1:n, (n+1):(n+lg_star));
Lambda_21=Lambda((n+1):(n+lg_star), 1:n);
Lambda_22=Lambda((n+1):(n+lg_star), (n+1):(n+lg_star));


L_Lb=jitterchol(Lambda_11);
v_Lb=L_Lb\Lambda_12;


Lambda_sig=Lambda_22-v_Lb'*v_Lb;

mean_dX=(L_K3'\(L_K3\KK2))';                                                % A matrix
cov_dX=cov_X-mean_dX*KK2;                                                   % B matrix                                                                               


L_dX=jitterchol(cov_dX);

    %--------------Draw Zd values from its marginal distribution------------
M=501;
Zd=zeros(set_num,M);                                                               % indicate Z'(s)
Zd_p=zeros(set_num,M);                                                             % indicate Z'^{+}(s) 

for k=2:M
%--------------Gibbs sampling procedure-----------------------------------
mu_m=zeros(set_num,1);
nu_m=zeros(set_num,1);

theta_m=zeros(set_num,1);
delta_m=zeros(set_num,1);

for i=1:set_num
%------------obtain the conditional proposal distribution------------------


%------------unconstraint posterior proposal distribution------------------
    
    temp_s=K3(i,:);
    temp_s(i)=[];
    temp_S=K3;
    temp_S(i,:)=[];
    temp_S(:,i)=[];

    
%------------unconstraint prior proposal distribution----------------------
    
    temp_a=mean_dX(:,i);

    if set_num~=1
       if i==1
          temp_A=mean_dX(:, (i+1:set_num));
       elseif i==set_num
          temp_A=mean_dX(:,(1:i-1));
       else
          temp_A=[mean_dX(:,(1:i-1)), mean_dX(:, (i+1:set_num))];
       end
    end
    
    if set_num~=1
       if i>1 && i<set_num
          temp_zp=[Zd((1:i-1),k);Zd((i+1:set_num),k-1)];
          temp_zpp=[Zd_p((1:i-1),k);Zd_p((i+1:set_num),k-1)];
          temp_zpp0=[Zd_p((1:i-1),k);0;Zd_p((i+1:set_num),k-1)];
        elseif i==1
          temp_zp=Zd(2:set_num,k-1);
          temp_zpp=Zd_p(2:set_num,k-1);
          temp_zpp0=[0;Zd_p(2:set_num,k-1)];
       else
          temp_zp=Zd(1:set_num-1,k);
          temp_zpp=Zd_p(1:set_num-1,k);
          temp_zpp0=[Zd_p(1:set_num-1,k);0];
       end

    end

   %--------------normal distribution for prior---------------------  
   if set_num~=1
    % inv_temp2=jitterinv(temp_S);
    
    tempS_chol=jitterchol(temp_S);
    v_tempS=tempS_chol\(temp_s');   
    alpha_tempS=tempS_chol'\(tempS_chol\temp_zp);
    
    mu_m(i)=temp_s*alpha_tempS;
    nu_m(i)=K3(i,i)-v_tempS'*v_tempS;
    nu_sqrt=sqrt(nu_m(i));
   else
    mu_m(i)=0;
    nu_m(i)=K3;
    nu_sqrt=sqrt(nu_m(i));
   end
%----------unconstraint posterior normal distribution------------    

   if set_num~=1
    v_dX=L_dX\temp_a;
    temp_th1=(x-mu)-temp_A*temp_zpp;
    alpha_dX=L_dX'\(L_dX\temp_th1);
    temp_th=temp_a'*alpha_dX+mu_m(i)*nu_m(i)^(-1);
    temp_dt1=nu_m(i)^(-1)+v_dX'*v_dX;
    temp_dt=1/temp_dt1;
    theta_m(i)=temp_dt*temp_th;
    delta_m(i)=sqrt(temp_dt);
   else
    v_dX=L_dX\mean_dX;
    alpha_dX=L_dX'\(L_dX\(x-mu));
    temp_th=mean_dX'*alpha_dX+mu_m(i)*nu_m(i)^(-1);
    temp_dt1=nu_m(i)^(-1)+v_dX'*v_dX;
    temp_dt=1/temp_dt1;
    theta_m(i)=temp_dt*temp_th;
    delta_m(i)=sqrt(temp_dt);
   end
%---------------Gibbs Sampling Procedure One--------------------

%---------------Binomial Selection Algorithm-------------------
  
    eps=0;    
    temp_ks=normcdf(eps,mu_m(i),nu_sqrt);

    temp_qx=1-normcdf(eps,theta_m(i),delta_m(i));
    
    
    temp_const=temp_qx*delta_m(i)/nu_sqrt*exp(-mu_m(i)^2/(2*nu_m(i))+theta_m(i)^2/(2*temp_dt));
    q_const=temp_ks/(temp_ks+temp_const);
    
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
    
    if biv_num==0
       u=temp_ks*rand(1);
       Zd(i,k)=norminv(u,mu_m(i),nu_sqrt);
       Zd_p(i,k)=0;
    else
       u=1-(temp_qx-(temp_qx)*rand(1));
       Zd(i,k)=norminv(u,theta_m(i),delta_m(i));
       Zd_p(i,k)=Zd(i,k);
    end
  
   
end

end



if (nstar-set_num)>0

alpha_K3=L_K3'\(L_K3\mean(Zd_p(:,51:M),2)); 
MU=mu_lam+Sigma_12*alpha_K3;

%---------------Compute Mean and Covariance-----------------------------

mu_z=MU((n+1):(n+lg_star));
mu_x=MU(1:n);

Z_mu=mu_z+Lambda_21*(L_Lb'\(L_Lb\(x-mu_x)));
Z_sigma=Lambda_sig/(M-50);


prob_y=normcdf(zeros(nstar-set_num,1),Z_mu,diag(sqrt(Z_sigma)));
else
 t2=tstar_g;
 break;
end
end

% ============================================================
% 3) 大 M Gibbs 阶段（与你第一段脚本一致）
% ============================================================
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



% ============================================================
% 4) 汇总样本并计算指标（潜在函数空间；与 IP 口径一致）
% ============================================================
Zval = Z(:, 50001:100000);              % 与脚本一致的 burn-in / 截取
Ef   = median(Zval, 2);
Varf = var(Zval, 0, 2);

% 95% 可信区间（用分位数，与脚本一致）
ci    = quantile(Zval', [0.025,0.975]);
ci_lo = ci(1,:)';
ci_hi = ci(2,:)';

% 评估（latent space）
epsv = 1e-12;

mse      = mean( (Ef - y_true).^2 );
nlpd     = -mean( log( max(realmin, normpdf(y_true, Ef, sqrt(max(Varf,epsv)))) ) );
coverage = mean( (y_true >= ci_lo) & (y_true <= ci_hi) );
width    = mean( ci_hi - ci_lo );

elapsedTime = toc(tAll);
figure;
plot(tstar,Ef,'.b','MarkerSize', 17);
hold on;
plot(tstar,y_true,'.r','MarkerSize', 17);
plot(tstar,ci_lo,'-.b','LineWidth', 1, 'MarkerSize', 17);
plot(tstar,ci_hi,'-.b','LineWidth', 1, 'MarkerSize', 17);
legend('Posterior Mean', 'Real Values', 'Lower 95% Credible Band', 'Upper 95% Credible Band', 'Location', 'Best');
hold off;  

end
