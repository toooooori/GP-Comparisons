gpml_root = 'gpml';
run(fullfile(gpml_root, 'startup.m'));
clear; clc; close all; rng(42);

n  = 50;
x1 = rand(n,1);
x2 = rand(n,1);
y  = [x1 x2];

%f_true = @(u,v) 1./(1+exp(-10*(u-0.4))) + 1./(1+exp(-10*(v-0.6)));
f_true = @(u,v) exp(u) + v + sin(20*v - 10)./20;
sigma   = 0.5;
epsilon = sigma * randn(n,1);
X       = f_true(x1, x2) + epsilon;

g = linspace(0,1,100);
[yt1, yt2] = meshgrid(g, g);
yt = [yt1(:) yt2(:)];
nstar = size(yt,1);

meanfunc = {@meanConst};   hyp.mean = 0;
covfunc  = {@covSEard};    hyp.cov  = [0, 0, 0];
likfunc  = {@likGauss};    hyp.lik  = -inf;

[hyp_sol, ~, ~] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, [], y, X);

dj = 2;
tAll = tic;
yt_s = CGP_BS_2D(hyp_sol, X, y, yt, dj);

Zval = Gibbs_CGP_2D(hyp_sol, X, y, yt, yt_s, dj);
validCols = all(~isnan(Zval), 1);
Zval = Zval(:, validCols); 

mu_cgp = mean(Zval, 2);

true_on_grid = f_true(yt1, yt2);
mse = mean((mu_cgp - true_on_grid(:)).^2);
elapsedTime = toc(tAll);

fprintf('Test-grid MSE = %.6f\n', mse);
fprintf('Time (sec): %.4f ± %.4f\n', elapsedTime);
Mu = reshape(mu_cgp, size(yt1));
Real = reshape(true_on_grid, size(yt1));
figure;
mesh(yt1, yt2, Mu, 'FaceColor', 'b', 'EdgeColor', 'b');
hold on;
mesh(yt1, yt2, Real, 'FaceColor', 'r', 'EdgeColor', 'r');
hold off;
legend('Predicted mean', 'True function', 'Location', 'Best');


function Zval = Gibbs_CGP_2D(hyp_sol, X, y, yt, yt_s, dj)
% Gibbs_CGP_2D  — Gibbs sampler for 2D Constrained GP with derivative
%                  monotonicity constraints.
%
% Key accelerations:
%   (1) Cholesky of the predictive covariance (Lambda_sig) is computed
%       once outside the Gibbs loop.
%   (2) Conditional updates of derivative variables use a precomputed
%       precision matrix Q = K3^{-1} instead of repeatedly forming
%       submatrix Cholesky factorizations.
%
% Inputs:
%   hyp_sol : learned hyperparameters (from GPML minimize)
%   X       : training targets, size [n x 1]
%   y       : training inputs, size [n x d]
%   yt      : prediction grid inputs, size [nstar x d]
%   yt_s    : selected constraint locations (where derivative is forced >=0)
%   dj      : which input dimension the monotonic constraint applies to
%
% Output:
%   Zval    : posterior samples of f(yt) from the 2nd half of the Gibbs chain
%             (columns 501:1000)

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
temp_covdX = jitterchol(cov_dX);             % chol(B)

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

L_Lb      = jitterchol(Lambda_11);
v_Lb      = L_Lb \ Lambda_12;
Lambda_sig= Lambda_22 - v_Lb' * v_Lb;

% ---------- Derivative covariance K3 ----------
K3  = Kdd_ys;

% Precompute Cholesky of Lambda_sig once
L_sig = jitterchol(Lambda_sig);

% Precompute precision Q = inv(K3) once
R  = jitterchol(K3);            % K3 = R * R'
I_m= eye(m);
Q  = R' \ (R \ I_m);            % numerical inverse
Q  = (Q + Q')/2;                % symmetrize

% ---------------- Gibbs Sampling ----------------
M    = 1000;
Z    = zeros(nstar, M);         % samples of f(yt)
Zd   = zeros(m,     M);         % samples of derivative Z'(yt_s)
Zd_p = zeros(m,     M);         % truncated / projected derivative

mu_lam = hyp_sol.mean * ones(n + nstar, 1);

for k = 2:M

    mu_m    = zeros(m,1);       % conditional means for Zd(i)
    nu_m    = zeros(m,1);       % conditional variances for Zd(i)
    theta_m = zeros(m,1);       % posterior proposal mean
    delta_m = zeros(m,1);       % posterior proposal std

    % --- coordinate-wise Gibbs updates for derivative process Zd ---
    for i = 1:m
        % Conditional for Zd(i) | Zd(-i) via precision matrix Q
        Qi_i    = Q(i,i);
        Qi_rest = Q(i,:);  Qi_rest(i) = [];  % Q_{i,-i}

        if i > 1 && i < m
            z_rest  = [Zd(1:i-1, k); Zd(i+1:m, k-1)];
            z_restp = [Zd_p(1:i-1, k); Zd_p(i+1:m, k-1)];
        elseif i == 1
            z_rest  = Zd(2:m, k-1);
            z_restp = Zd_p(2:m, k-1);
        else
            z_rest  = Zd(1:m-1, k);
            z_restp = Zd_p(1:m-1, k);
        end

        mu_i    = -(Qi_rest / Qi_i) * z_rest;
        nu_i    =  1 / Qi_i;
        nu_sqrt = sqrt(nu_i);

        mu_m(i) = mu_i;
        nu_m(i) = nu_i;

        % Posterior "proposal" terms using cov_dX
        temp_a  = mean_dX(:, i);
        if i == 1
            temp_A = mean_dX(:, 2:m);
        elseif i == m
            temp_A = mean_dX(:, 1:m-1);
        else
            temp_A = [mean_dX(:, 1:i-1), mean_dX(:, i+1:m)];
        end

        v_dX     = temp_covdX \ temp_a;
        temp_th1 = (X - hyp_sol.mean) - temp_A * z_restp;
        alpha_dX = temp_covdX' \ (temp_covdX \ temp_th1);

        temp_th  = temp_a' * alpha_dX + mu_m(i) * (1/nu_m(i));
        temp_dt1 = (1/nu_m(i)) + (v_dX' * v_dX);
        temp_dt  = 1 / temp_dt1;

        theta_m(i) = temp_dt * temp_th;
        delta_m(i) = sqrt(temp_dt);

        % Truncated/mixture sampling to enforce monotonicity (Zd_p >= 0)
        eps_val  = 0;
        temp_ks  = normcdf(eps_val, mu_m(i), nu_sqrt);
        temp_qx  = 1 - normcdf(eps_val, theta_m(i), delta_m(i));

        temp_const = temp_qx * sqrt(delta_m(i)/nu_sqrt) * ...
            exp( -mu_m(i)^2/(2*nu_m(i)) + theta_m(i)^2/(2*temp_dt) );

        q_const = temp_const / (temp_ks + temp_const);
        if isinf(temp_const), q_const = 1; end
        if isnan(q_const)
            warning('NaN in mixture weight');
            q_const = 0.5;
        end
        if temp_ks == 0 && temp_const == 0
            temp_ks = 1e-6;
            q_const = temp_const / (temp_ks + temp_const);
        end

        % Bernoulli branch
        draw_flag = binornd(1, q_const);

        if draw_flag == 1
            % sample from truncated N(mu_m, nu_m) below 0
            u        = temp_ks * rand(1);
            Zd(i,k)  = norminv(u, mu_m(i), nu_sqrt);
            Zd_p(i,k)= 0;
        else
            % sample from truncated N(theta_m, delta_m) above 0
            u        = 1 - (temp_qx - temp_qx*rand(1));
            Zd(i,k)  = norminv(u, theta_m(i), delta_m(i));
            Zd_p(i,k)= Zd(i,k);
        end
    end

    % --- Conditional Gaussian for f(yt) given Zd_p ---
    alpha_K3 = L_ys' \ (L_ys \ Zd_p(:,k));
    MU       = mu_lam + Sigma_12 * alpha_K3;  % joint mean [f(y); f(yt)]

    mu_z = MU(n+1 : n+nstar);   % mean at yt
    mu_x = MU(1:n);             % mean at training inputs

    Z_mu = mu_z + Lambda_21 * (L_Lb' \ (L_Lb \ (X - mu_x)));

    % Draw one sample of f(yt) using precomputed Cholesky
    Z(:,k) = Z_mu + L_sig * randn(nstar, 1);
end

% Keep the latter half of samples (burn-in discarded)
Zval = Z(:, 501:1000);

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
