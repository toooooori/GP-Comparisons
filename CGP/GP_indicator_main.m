%----------------CopyRight @ Xiaojing Wang----------------------------
GPsol_prob

n =length(t1);
t2=sort(t2);                 % derivative points
m =length(t2);
nstar=100;


covfunc = {'covSum', {'covSEiso','covNoise'}};


M=10000;
Z=zeros(nstar,M);
Zd=zeros(m,M);
tstar=linspace(-10,10,nstar)';



cov_X=feval(covfunc{:}, loghyper, t1);
cov_ZT=covSEiso(loghyper(1:2),t1);                    % prior covariance matrix of Z(T)
[K1, K2, K3, KK2]=covGP(loghyper(1:2), t1, t2);


figure;
disp('  plot(t, x, ''k+'')')
plot(t1, x, 'k+', 'MarkerSize', 17)
disp('  hold on')
hold on
jitter=1e-6;

chol_covX=jitterchol(cov_X);
mu_dS=KK2*(chol_covX'\(chol_covX\(x-mu)));
v_covX=chol_covX\K2;
cov_dS=K3-v_covX'*v_covX;

chol_K3=jitterchol(K3);
inv_K3=chol_K3'\(chol_K3\eye(size(K3,1)));
v_K3=chol_K3\KK2;


[Ks1, Ks2, Ks3, KKs2]=covGP(loghyper(1:2), tstar, t2);
v_Ks3=chol_K3\KKs2;


cov_Zxt=covSEiso(loghyper(1:2),tstar,t1);                    % prior covariance matrix of Z(T)

B_1=cov_X-v_K3'*v_K3;
B_2=Ks1-v_Ks3'*v_Ks3;
B_3=cov_Zxt-v_Ks3'*v_K3;

chol_B1=jitterchol(B_1);
v_B1=chol_B1\(B_3');

mu_z=mu*ones(nstar,1)+v_B1'*(chol_B1\(x-mu*ones(n,1)));
Z_sigma=B_2-v_B1'*v_B1;
temp_coef=Ks2-v_B1'*(chol_B1\K2);
mu_coef=chol_K3\(temp_coef');

for k=2:M

mu_S=zeros(m,1);
sigma_S=zeros(m,1);

%--------------Gibbs sampling procedure-----------------------------------
for i=1:m

    if i==2
        a=1;
    end
%------------obtain the conditional proposal distribution------------------

    temp_s=cov_dS(i,:);
    temp_s(i)=[];
    temp_S=cov_dS;
    temp_S(i,:)=[];
    temp_S(:,i)=[];
    if i>1 && i<m
    temp_zs=[Zd((1:i-1),k);Zd((i+1:m),k-1)];
    elseif i==1
        temp_zs=Zd(2:m,k-1);
    else
        temp_zs=Zd(1:m-1,k);
    end
    
        mu_dSi=mu_dS;
    if (m>1)
         mu_dSi(i)=[];
    end
    
    chol_tempS=jitterchol(temp_S);
    alpha_tempS=chol_tempS'\(chol_tempS\(temp_zs-mu_dSi));
    v_tempS=chol_tempS\(temp_s');
    
    mu_S(i)=mu_dS(i)+temp_s*alpha_tempS;
    sigma_S(i)=sqrt(cov_dS(i,i)-v_tempS'*v_tempS);

%---------------Gibbs Sampling Procedure One--------------------
%---------------Truncated Normal Distribution-------------------

    temprand=normcdf(0,mu_S(i),sigma_S(i));
    u=temprand+(1-temprand)*rand(1);
    Zd(i,k)=norminv(u,mu_S(i),sigma_S(i));
    if Zd(i,k)==Inf
       Zd(i,k)=0;
       warning('Zd values resetted at' );
       warning(num2str(i,k));
    end
    
end


%---------------Gibbs Sampling Procedure Two--------------------
%---------------Multivariate Normal Distribution----------------

Z_mu=mu_z+mu_coef'*(chol_K3\Zd(:,k));

Z(:,k)=Z_mu+jitterchol(Z_sigma)*randn(nstar,1);


end

if fm==1
    xstar=sin(tstar)+tstar;
elseif fm==2
    xstar=4./(1+exp(-tstar./2+4)); 
elseif fm==3
    
      ystar=zeros(nstar,1);
      for i=1:nstar
          if tstar(i)<-5
             ystar(i)=tstar(i)+5;
          elseif tstar(i)>5
             ystar(i)=tstar(i)-5;
          else
             ystar(i)=0;
          end
     end

   xstar=ystar;
end

Zval=Z(:,5001:10000);  
Z_qt=quantile(Zval', [0.025,0.975]);  
Z_qt1=Z_qt(1,:);
Z_qt2=Z_qt(2,:);

if (m>1)
Zd_val=Zd(:,5001:10000);
Zd_qt=quantile(Zd_val', [0.025,0.975]);  
Zd_qt1=Zd_qt(1,:);
Zd_qt2=Zd_qt(2,:);

end
figure;
plot(tstar,median(Zval,2),'.b','MarkerSize', 17);
hold on;
plot(tstar,xstar,'.r','MarkerSize', 17);
plot(tstar,Z_qt1,'-.b','LineWidth', 1, 'MarkerSize', 17);
plot(tstar,Z_qt2,'-.b','LineWidth', 1,'MarkerSize', 17);

plot(t1,x,'+k','MarkerSize', 13)
ay=axis;

plot(t2,ay(3),'.g','MarkerSize', 17) 


    h = legend('Posterior Median','Real Values',...
        'Upper 95% Credible Band', 'Lower 95% Credible Band','Observations','Gridded Points', 2);
    set(h,'Interpreter','none')
