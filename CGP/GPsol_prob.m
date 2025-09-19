%------------CopyRight @ Xiaojing Wang-----------------------

startup

%------------simulation GP data-------------------------------
n =50;
nstar=100;

sigma_n=sqrt(0.25);

t1=sort( 20*(rand(n,1)-0.5));

ys=zeros(n,1);
for i=1:n
if t1(i)<-5
ys(i)=t1(i)+5;
elseif t1(i)>5
ys(i)=t1(i)-5;
else
ys(i)=0;
end
end

fm=3;

if fm==1
    x=4./(1+exp(-t1./2+4))+sigma_n*randn(n,1);
elseif fm==2
    x=sin(t1)+t1+sigma_n*randn(n,1);
elseif fm==3
x=ys+sigma_n*randn(n,1);
end

tstar=linspace(-10,10,nstar)';

meanfunc ={@meanConst}; hyp.mean = 0;

covfunc = {'covSum',{'covSEisoS', 'covNoise'}};  hyp.cov = [0, 0, 0];
likfunc = {@likGauss};  hyp.lik = -Inf;

[hyp_sol, f, i] = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, [], t1, x);

loghyper=hyp_sol.cov;
ell=exp(hyp_sol.cov(1));
sigma_f=exp(hyp_sol.cov(2));
mu=hyp_sol.mean;

cov_X=feval(covfunc{:}, loghyper, t1);
cov_ZT=covSEiso([log(ell);log(sigma_f)],t1);                    % prior covariance matrix of Z(T)

%---------------------Find Maximum Points-----------------------------

%---------------------Initial Start Conditions-----------------------
set_num=0;
[Ks1,Ks2,Ks3,KKs2]=covGP([log(ell);log(sigma_f)],t1,tstar);

L_covX=jitterchol(cov_X);
alpha_covX=L_covX'\(L_covX\(x-mu));
mu_dS=KKs2*alpha_covX;
v_covX=L_covX\Ks2;
cov_dS=Ks3-v_covX'*v_covX;

prob_y=normcdf(zeros(nstar-set_num,1),mu_dS,diag(sqrt(cov_dS)));

figure;
plot(tstar,prob_y,'.','MarkerSize', 17)

th_hd=1e-200;
t2=[];
tstar_g=tstar;

while max(prob_y)>th_hd

set_num=set_num+1;    

temp=find(prob_y==max(prob_y));
lg=length(temp);
if lg~=1
   a_mat=temp(1);
   temp_idx=0;
   for i=1:lg
       if temp(i)~=(a_mat+i-1) 
          temp_idx=i-1;
          break;
       end
   end    
   if temp_idx~=0;
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

mu_dS=KK2*alpha_covX;
temp_vcov=L_covX\K2;
cov_dS=K3-temp_vcov'*temp_vcov;

mu_S=zeros(set_num,1);
sigma_S=zeros(set_num,1);

M=501;
Zd=zeros(set_num,M);

for k=2:M
%--------------Gibbs sampling procedure-----------------------------------
mu_S=zeros(set_num,1);
sigma_S=zeros(set_num,1);

for i=1:set_num

    
%------------obtain the conditional proposal distribution------------------
    if set_num~=1
    temp_s=cov_dS(i,:);
    temp_s(i)=[];
    temp_S=cov_dS;
    temp_S(i,:)=[];
    temp_S(:,i)=[];
    if i>1 && i<set_num
    temp_zs=[Zd((1:i-1),k);Zd((i+1:set_num),k-1)];
    elseif i==1
        temp_zs=Zd(2:set_num,k-1);
    else
        temp_zs=Zd(1:set_num-1,k);
    end
             
    mu_dSi=mu_dS;
    mu_dSi(i)=[];

    tempS_chol=jitterchol(temp_S);
    temp_alpha=tempS_chol'\(tempS_chol\(temp_zs-mu_dSi));
    
    mu_S(i)=mu_dS(i)+temp_s*temp_alpha;
    
    tempS_v=tempS_chol\(temp_s');

    sigma_S(i)=sqrt(cov_dS(i,i)-tempS_v'*tempS_v);
    else
     mu_S(i)=mu_dS;
     sigma_S(i)=sqrt(cov_dS);
    end
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
    
%-----------


end
end



if nstar-set_num>0
   Zd_temp=mean(Zd(:,(51:M)),2); 
   tstar=setdiff(tstar,t2);
   
   tempK2=gp_dist(t2'/ell,tstar'/ell);
   K_3 = sigma_f^2*exp(-tempK2.^2/2).*1/ell^2.*(1-tempK2.^2);

   
   [Ks1,Ks2,Ks3,KKs2]=covGP([log(ell);log(sigma_f)],t1,tstar);
   
   chol_K3=jitterchol(K3);
   inv_K3=chol_K3'\(chol_K3\eye(size(K3,1)));
   v_K3=chol_K3\KK2;
   vv_K3=chol_K3\K_3;
   
   A_1=K2*inv_K3;
   A_2=K_3'*inv_K3;
   
   B_1=cov_X-v_K3'*v_K3;
   B_2=Ks3-vv_K3'*vv_K3;
   B_3=KKs2-(K_3')*inv_K3*KK2;
   
   chol_B1=jitterchol(B_1);
   
   v_B1=chol_B1\(B_3');
   alpha_B1=chol_B1'\(chol_B1\(x-mu));
   alpha_BA=chol_B1'\(chol_B1\A_1);

 
   mu_dS=B_3*alpha_B1+(A_2-B_3*alpha_BA)*Zd_temp;
   cov_dS=(B_2-v_B1'*v_B1)/(M-50);
   

   prob_y=normcdf(zeros(nstar-set_num,1),mu_dS,diag(sqrt(cov_dS)));

   figure;
   plot(tstar,prob_y,'.', 'MarkerSize', 17)
   
else
   t2=tstar_g;
 break;
end
end




