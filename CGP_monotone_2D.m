gpml_root = 'gpml';
run(fullfile(gpml_root, 'startup.m'));
clear; clc; close all;

rng(42,'twister');

n  = 50;
x1 = rand(n,1);
x2 = rand(n,1);
y  = [x1 x2];

f_true = @(u,v) 1./(1+exp(-10*(u-0.4))) + 1./(1+exp(-10*(v-0.6)));
%f_true = @(u,v) exp(u) + v + sin(20*v - 10)./20;
sigma   = 0.5;
epsilon = sigma * randn(n,1);
X       = f_true(x1, x2) + epsilon;

g_cand = linspace(0,1,50);        
[yc1, yc2] = meshgrid(g_cand, g_cand);
yt_cand = [yc1(:) yc2(:)];

g_pred = linspace(0,1,50);
[yp1, yp2] = meshgrid(g_pred, g_pred);
yt_pred = [yp1(:) yp2(:)];

meanfunc = {@meanConst};   hyp.mean = 0;
covfunc  = {@covSEard};    hyp.cov  = [0, 0, 0];
likfunc  = {@likGauss};    hyp.lik  = -inf;

[hyp_sol, ~, ~] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, [], y, X);

dj = 2;
tAll = tic;
yt_s = CGP_BS_2D(hyp_sol, X, y, yt_cand, dj);

[mu_cgp, var_pt] = Gibbs_CGP_2D_mean(hyp_sol, X, y, yt_pred, yt_s, dj);

true_on_grid = f_true(yp1, yp2);
mse = mean((mu_cgp - true_on_grid(:)).^2);
elapsedTime = toc(tAll);

fprintf('Test-grid MSE = %.6f\n', mse);
fprintf('Time (sec): %.4f\n', elapsedTime);

Mu   = reshape(mu_cgp, size(yp1));
Real = true_on_grid;

figure;
mesh(yp1, yp2, Mu, 'FaceColor', 'b', 'EdgeColor', 'b'); hold on;
mesh(yp1, yp2, Real, 'FaceColor', 'r', 'EdgeColor', 'r'); hold off;
legend('Predicted mean', 'True function', 'Location', 'Best');



function [mu_hat, var_pt] = Gibbs_CGP_2D_mean(hyp_sol, X, y, yt, yt_s, dj)
% Gibbs_CGP_2D_mean — 2D CGP with derivative monotonicity constraints:
%                      compute ONLY the posterior mean (optionally pointwise variance).
%
% Key idea: keep the Gibbs on derivative variables Z' (dimension m, small),
%           but DO NOT sample the full field f(yt). Instead, at each
%           iteration compute the closed-form conditional mean Z_mu and
%           average across iterations. This avoids the (nstar x nstar)
%           Cholesky and per-iteration triangular multiplies.

% ---------------- Sizes ----------------
n     = size(y, 1);
nstar = size(yt, 1);
m     = size(yt_s, 1);

% ---------- Covariances for derivative process Z'(yt_s) ----------
[K_yy, Kd_yys, Kdd_ys, Kd_ysy] = covSEard_GP(hyp_sol.cov, y, yt_s, dj);
L_ys  = jitterchol(Kdd_ys);

% Posterior of derivative process Z'(·)
mean_dX    = (L_ys' \ (L_ys \ Kd_ysy))';     % A
cov_dX     = K_yy - mean_dX * Kd_ysy;        % B
L_dX       = jitterchol(cov_dX);             % chol(B) for stable solves

% ---------- Joint covariances for f(y) and f(yt) ----------
[K_ytyt, Kd_ytys, ~, Kd_ysyt] = covSEard_GP(hyp_sol.cov, yt, yt_s, dj);
K_yyt = covSEardS(hyp_sol.cov, y, yt);

Sigma_11 = [K_yy,   K_yyt;
            K_yyt', K_ytyt];
Sigma_12 = [Kd_yys;
            Kd_ytys];
Sigma_21 = [Kd_ysy, Kd_ysyt];

v_ys   = L_ys \ Sigma_21;
Lambda = Sigma_11 - v_ys' * v_ys;

Lambda_11 = Lambda(1:n,           1:n);
Lambda_12 = Lambda(1:n,           n+1:n+nstar);
Lambda_21 = Lambda(n+1:n+nstar,   1:n);
Lambda_22 = Lambda(n+1:n+nstar,   n+1:n+nstar);

L_Lb   = jitterchol(Lambda_11);
v_Lb   = L_Lb \ Lambda_12;

% pointwise variance (optional): diag(Lambda_sig) without forming chol
Lambda_sig_diag = diag(Lambda_22) - sum(v_Lb.^2, 1)';  % equals diag(Lambda_22 - v_Lb'*v_Lb)

% ---------- Derivative covariance precision Q ----------
K3  = Kdd_ys;
R   = jitterchol(K3);                  % K3 = R R'
I_m = eye(m);
Q   = R' \ (R \ I_m);                  % numerical inverse (precision)
Q   = (Q + Q')/2;                      % symmetrize

% ---------------- Gibbs on Z' (small m) ----------------
M       = 10000;
burnin  = 5000;
sum_mu  = zeros(nstar,1);
nsamp   = 0;

mu_lam  = hyp_sol.mean * ones(n + nstar, 1);
Zd      = zeros(m, 1);                 % can be warm-started if desired
Zd_p    = max(Zd, 0);

for k = 2:M
    % --- coordinate-wise Gibbs updates for derivative process Zd ---
    for i = 1:m
        % Conditional for Zd(i) | Zd(-i) via precision matrix Q
        Qi_i    = Q(i,i);
        Qi_rest = Q(i,:);  Qi_rest(i) = [];

        if i > 1 && i < m
            z_rest  = [Zd(1:i-1); Zd(i+1:m)];
            z_restp = [Zd_p(1:i-1); Zd_p(i+1:m)];
        elseif i == 1
            z_rest  = Zd(2:m);
            z_restp = Zd_p(2:m);
        else
            z_rest  = Zd(1:m-1);
            z_restp = Zd_p(1:m-1);
        end

        mu_i    = -(Qi_rest / Qi_i) * z_rest;
        nu_i    =  1 / Qi_i;
        nu_sqrt = sqrt(nu_i);

        % Posterior "proposal" terms using L_dX (chol of cov_dX)
        temp_a  = mean_dX(:, i);
        if i == 1
            temp_A = mean_dX(:, 2:m);
        elseif i == m
            temp_A = mean_dX(:, 1:m-1);
        else
            temp_A = [mean_dX(:, 1:i-1), mean_dX(:, i+1:m)];
        end

        v_dX     = L_dX \ temp_a;
        temp_th1 = (X - hyp_sol.mean) - temp_A * z_restp;
        alpha_dX = L_dX' \ (L_dX \ temp_th1);

        temp_th  = temp_a' * alpha_dX + mu_i * (1/nu_i);
        temp_dt1 = (1/nu_i) + (v_dX' * v_dX);
        temp_dt  = 1 / temp_dt1;

        theta_i  = temp_dt * temp_th;
        delta_i  = sqrt(temp_dt);

        % Mixture truncation to enforce Zd_p >= 0
        eps_val  = 0;
        temp_ks  = normcdf(eps_val, mu_i,    nu_sqrt);
        temp_qx  = 1 - normcdf(eps_val, theta_i, delta_i);

        temp_const = temp_qx * sqrt(delta_i/nu_sqrt) * ...
            exp( -mu_i^2/(2*nu_i) + theta_i^2/(2*temp_dt) );

        q_const = temp_const / (temp_ks + temp_const);
        if isinf(temp_const), q_const = 1; end
        if isnan(q_const),    q_const = 0.5; end
        if temp_ks == 0 && temp_const == 0
            temp_ks = 1e-6; q_const = temp_const / (temp_ks + temp_const);
        end

        if rand < q_const
            % sample from truncated N(mu_i, nu_i) below 0
            u      = temp_ks * rand(1);
            Zd(i)  = norminv(u, mu_i, nu_sqrt);
            Zd_p(i)= 0;
        else
            % sample from truncated N(theta_i, delta_i) above 0
            u      = 1 - (temp_qx - temp_qx * rand(1));
            Zd(i)  = norminv(u, theta_i, delta_i);
            Zd_p(i)= Zd(i);
        end
    end

    % --- Closed-form conditional mean for f(yt) given current Zd_p ---
    alpha_K3 = L_ys' \ (L_ys \ Zd_p);
    MU       = mu_lam + Sigma_12 * alpha_K3;  % joint mean [f(y); f(yt)]

    mu_z = MU(n+1 : n+nstar);   % mean at yt
    mu_x = MU(1:n);             % mean at training inputs

    Z_mu = mu_z + Lambda_21 * (L_Lb' \ (L_Lb \ (X - mu_x)));

    % --- Accumulate mean after burn-in (NO full-field sampling) ---
    if k > burnin
        if all(isfinite(Z_mu))
        sum_mu = sum_mu + Z_mu;
        nsamp  = nsamp + 1;
        else
        end
    end
end

mu_hat = sum_mu / max(nsamp, 1);
var_pt = max(Lambda_sig_diag, 0);  % pointwise variance (optional)
end


function yt_s = CGP_BS_2D(hyp_sol, X, y, yt, dj)
% CGP_BS_2D
% Selects a sparse set of constraint locations yt_s in 2D for a
% monotonic-constrained GP.
%
% This routine:
%   1. Computes the GP posterior over the derivative process at candidate
%      grid points yt.
%   2. Iteratively picks the worst-violation point (most likely to break
%      monotonicity).
%   3. Runs a short Gibbs sampler over the derivative at the selected
%      constraint set to estimate updated violation at the remaining
%      candidates.
%   4. Rejects adding new constraint points that are too close to already
%      selected ones (min_dist_tol) to avoid redundant support points.
%
% Inputs:
%   hyp_sol : learned hyperparameters (from GPML minimize)
%   X       : training targets, size [n x 1]
%   y       : training inputs, size [n x d]
%   yt      : candidate grid locations for constraints (nstar x d)
%   dj      : dimension index of monotonicity constraint
%
% Output:
%   yt_s    : selected constraint set (subset of yt)

[K_yy, Kd_yystar, Kdd_ystar, Kd_ystary] = covSEard_GP(hyp_sol.cov, y, yt, dj);

L_yy   = jitterchol(K_yy);
X_mu   = X - hyp_sol.mean;
alpha_mu = L_yy' \ (L_yy \ X_mu);

mu_Kdd  = Kd_ystary * alpha_mu;
v_mu    = L_yy \ Kd_yystar;
cov_Kdd = Kdd_ystar - v_mu' * v_mu;

nstar = size(yt,1);

set_num = 0;
prob_y  = normcdf(zeros(nstar - set_num,1), mu_Kdd, diag(sqrt(cov_Kdd)));

th_hd   = 1e-200;
yt_s    = [];
yt_starg= yt;
a       = []; %#ok<NASGU>  % tracking Inf fixes (debug only)

% minimum distance tolerance so we don't add near-duplicate constraint points
min_dist_tol = 1e-3;

while max(prob_y) > th_hd

    set_num = set_num + 1;

    % pick the location with max violation probability
    temp = find(prob_y == max(prob_y));
    lg   = length(temp);

    if lg ~= 1
        a_mat    = temp(1);
        temp_idx = 0;
        for i = 1:lg
            if temp(i) ~= (a_mat + i - 1)
                temp_idx = i-1;
                break;
            end
        end    
        if temp_idx ~= 0
            if mod(temp_idx,2)==0
                temp_va = yt(temp(temp_idx/2),:);
            else
                temp_va = yt(temp((temp_idx+1)/2),:);
            end
        else
            if mod(lg,2)==0
                temp_va = yt(temp(lg/2),:);
            else
                temp_va = yt(temp((lg+1)/2),:);
            end
        end
    else
        temp_va = yt(temp,:);
    end

    % reject points that are too close to any already selected constraint
    if ~isempty(yt_s)
        d2 = sum((yt_s - temp_va).^2, 2);
        if any(d2 < min_dist_tol^2)
            % undo the increment because we didn't actually add a new point
            set_num = set_num - 1;

            % drop near-duplicates of this candidate from yt (so we don't pick it again)
            keep_mask = true(size(yt,1),1);
            for rr = 1:size(yt,1)
                if sum((yt(rr,:) - temp_va).^2) < min_dist_tol^2
                    keep_mask(rr) = false;
                end
            end
            yt = yt(keep_mask,:);

            % shrink mu_Kdd / cov_Kdd / prob_y consistently
            mu_Kdd  = mu_Kdd(keep_mask,:);
            cov_Kdd = cov_Kdd(keep_mask, keep_mask);

            prob_y = normcdf( ...
                zeros(size(mu_Kdd,1),1), ...
                mu_Kdd, ...
                diag(sqrt(cov_Kdd)) );

            % continue to next loop iteration with updated candidate pool
            continue;
        end
    end

    % accept this constraint point
    yt_s = [yt_s; temp_va];

    % recompute posterior of derivative at selected constraint set
    [K_yy, Kd_yyts, Kdd_yts, Kd_ytsy] = covSEard_GP(hyp_sol.cov, y, yt_s, dj);

    mu_Kddnew  = Kd_ytsy * alpha_mu;
    v_munew    = L_yy \ Kd_yyts;
    cov_Kddnew = Kdd_yts - v_munew' * v_munew;

    % short Gibbs chain to approximate derivative distribution at yt_s
    M  = 501;
    Zd = zeros(set_num, M);

    for k = 2:M
        mu_S    = zeros(set_num,1);
        sigma_S = zeros(set_num,1);

        for ii = 1:set_num

            if set_num ~= 1
                temp_s      = cov_Kddnew(ii,:);
                temp_s(ii)  = [];
                temp_S      = cov_Kddnew;
                temp_S(ii,:)= [];
                temp_S(:,ii)= [];

                if ii>1 && ii<set_num
                    temp_zs = [Zd(1:ii-1,k); Zd(ii+1:set_num,k-1)];
                elseif ii==1
                    temp_zs = Zd(2:set_num,k-1);
                else
                    temp_zs = Zd(1:set_num-1,k);
                end

                mu_dSi        = mu_Kddnew;
                mu_dSi(ii)    = [];

                tempS_chol = jitterchol(temp_S);
                temp_alpha = tempS_chol' \ (tempS_chol \ (temp_zs - mu_dSi));
                mu_S(ii)   = mu_Kddnew(ii) + temp_s * temp_alpha;

                tempS_v     = tempS_chol \ (temp_s');
                sigma_S(ii) = sqrt(cov_Kddnew(ii,ii) - tempS_v' * tempS_v);
            else
                mu_S(ii)    = mu_Kddnew;
                sigma_S(ii) = sqrt(cov_Kddnew);
            end

            % truncated normal draw enforcing derivative >= 0
            temprand = normcdf(0, mu_S(ii), sigma_S(ii));
            u        = temprand + (1-temprand)*rand(1);
            Zd(ii,k) = norminv(u, mu_S(ii), sigma_S(ii));

            if Zd(ii,k) == Inf || Zd(ii,k) == -Inf
                Zd(ii,k) = 0;
            end
        end
    end

    if nstar - set_num > 0
        % posterior mean of derivative after burn-in
        Zd_temp = mean(Zd(:, 51:M), 2);

        % remaining candidate pool
        yt_sl = yt_starg(~ismember(yt_starg, yt_s,'rows')', :);

        % joint Gaussian over [Z'(yt_s); X]
        Kcov      = [Kdd_yts,  Kd_ytsy;
                     Kd_yyts,  K_yy];
        Kcov_chol = jitterchol(Kcov);

        Kmean      = [Zd_temp; X - hyp_sol.mean];
        alpha_Kcov = Kcov_chol' \ (Kcov_chol \ Kmean);

        % predictive derivative posterior at remaining points
        Kdd_ytslys = covSEard_GPD(hyp_sol.cov, yt_sl, yt_s, dj);
        Kdd_ysytsl = covSEard_GPD(hyp_sol.cov, yt_s, yt_sl, dj);

        [K_yy, Kd_yytsl, Kdd_ytsl, Kd_ytsly] = covSEard_GP(hyp_sol.cov, y, yt_sl, dj);

        temp_list = [Kdd_ytslys,  Kd_ytsly];
        temp_up   = [Kdd_ysytsl;  Kd_yytsl];

        mu_Kl  = temp_list * alpha_Kcov;
        v_Kl   = Kcov_chol \ (temp_up);
        cov_Kl = Kdd_ytsl - v_Kl' * v_Kl;
        cov_Kl = cov_Kl / (M - 50);

        prob_y = normcdf( ...
            zeros(nstar - set_num,1), ...
            mu_Kl, ...
            diag(sqrt(cov_Kl)) );

        yt = yt_sl; % shrink candidate set

    else
        yt_s = yt_starg;
        break;
    end

end

end
