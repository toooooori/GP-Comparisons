function [mse, nlpd, coverage, width, elapsedTime] = CGP_convex(t1_in, x_in, tstar_in, y_true)
% CGP_CONVEX
% Constrained Gaussian Process regression under convexity (f'' >= 0).
%
% Output:
%   mse          - mean squared error on test grid
%   nlpd         - negative log predictive density
%   coverage     - empirical 95% credible interval coverage
%   width        - average 95% credible interval width
%   elapsedTime  - wall time in seconds
%
% Dependencies:
%   GPML toolbox

% ----------- I/O reshaping -----------
t1    = t1_in(:);      % training inputs (n x 1)
x     = x_in(:);       % training outputs (n x 1)
tstar = tstar_in(:);   % test grid (n* x 1)
n     = numel(t1);
nstar = numel(tstar);

tAll = tic;

% ============================================================
% 1) Fit GP hyperparameters (unconstrained GP)
% ============================================================
meanfunc = {@meanConst};             hyp.mean = 0;
covfunc  = {'covSum',{'covSEisoS','covNoise'}};
hyp.cov  = [0, 0, 0];                % [log(ell); log(sf); log(sn)]
likfunc  = {@likGauss};
hyp.lik  = -Inf;

[hyp_sol, ~, ~] = minimize(hyp, @gp, -500, @infExact, ...
    meanfunc, covfunc, [], t1, x);

loghyper = hyp_sol.cov;
ell      = exp(hyp_sol.cov(1));
sigma_f  = exp(hyp_sol.cov(2));
mu       = hyp_sol.mean;

cov_X  = feval(covfunc{:}, loghyper, t1);     % K(f(t1), f(t1))

% ============================================================
% 2) Active set selection of curvature constraints (f'' >= 0)
% ============================================================
set_num = 0;

[Ks1, Ks2, Ks3, KKs2] = covGP2([log(ell);log(sigma_f)], t1, tstar);
% Ks2   = Cov(f(t1), f''(tstar))
% KKs2  = Cov(f''(tstar), f(t1))
% Ks3   = Cov(f''(tstar), f''(tstar))

L_covX     = jitterchol(cov_X);
alpha_covX = L_covX' \ (L_covX \ (x - mu));

mu_dS  = KKs2 * alpha_covX;               % mean of f''(tstar)
v_covX = L_covX \ Ks2;
cov_dS = Ks3 - v_covX' * v_covX;          % cov of f''(tstar)

prob_y = normcdf( zeros(nstar - set_num,1), ...
                   mu_dS, ...
                   diag(sqrt(cov_dS)) );

th_hd   = 1e-20;
t2      = [];          % active curvature-constraint locations
tstar_g = tstar;       % save full grid

while max(prob_y) > th_hd

    set_num = set_num + 1;

    % pick location with max violation probability
    temp = find(prob_y == max(prob_y));
    lg   = length(temp);
    if lg ~= 1
        j = 1; temp_idx = 0;
        while j < lg
            a_mat = temp(j);
            for i = j:lg
                if temp(i) ~= (a_mat + i - j)
                    temp_idx = i - j; break;
                end
            end
            if temp_idx == 0
                j = i;
            else
                temp_idx = i - 1; break;
            end
        end
        if temp_idx ~= 0
            if mod(temp_idx,2)==0
                temp_va = tstar(temp(temp_idx/2));
            else
                temp_va = tstar(temp((temp_idx+1)/2));
            end
        else
            if mod(lg,2)==0
                temp_va = tstar(temp(lg/2));
            else
                temp_va = tstar(temp((lg+1)/2));
            end
        end
    else
        temp_va = tstar(temp);
    end

    % add to active set
    t2 = [t2; temp_va];

    % GP covariances wrt active constraint set t2
    [K1, K2, K3, KK2] = covGP2([log(ell);log(sigma_f)], t1, t2); %#ok<ASGLU>
    % K2  = Cov(f(t1), f''(t2))
    % KK2 = Cov(f''(t2), f(t1))
    % K3  = Cov(f''(t2), f''(t2))

    % remove chosen point(s) from candidate pool
    tstar    = setdiff(tstar, t2);
    lg_star  = length(tstar);

    % recompute covariances for remaining candidate points
    [Ks1, Ks2, Ks3, KKs2] = covGP2([log(ell);log(sigma_f)], t1, tstar); %#ok<ASGLU>

    % cross-kernel Cov(f''(t2), f''(tstar)) under squared exponential
    tempK2 = gp_dist(t2'/ell, tstar'/ell);
    K_3 = sigma_f^2 * exp(-tempK2.^2 / 2) .* (1/ell^4) .* ...
          (tempK2.^4 - 6*tempK2.^2 + 3);

    % joint GP blocks
    Sigma_11 = [cov_X, Ks2;
                KKs2,  Ks3];
    Sigma_12 = [K2;
                K_3'];
    Sigma_21 = [KK2, K_3];
    Sigma_22 = K3;

    % condition on curvature process f''(t2)
    L_K3 = jitterchol(K3);
    v_K3 = L_K3 \ Sigma_21;

    mu_lam = [mu * ones(n,1); zeros(lg_star,1)];
    Lambda = Sigma_11 - v_K3' * v_K3;

    Lambda_11 = Lambda(1:n,              1:n);
    Lambda_12 = Lambda(1:n,              n+1:n+lg_star);
    Lambda_21 = Lambda(n+1:n+lg_star,    1:n);
    Lambda_22 = Lambda(n+1:n+lg_star,    n+1:n+lg_star);

    L_Lb  = jitterchol(Lambda_11);
    v_Lb  = L_Lb \ Lambda_12;
    Lambda_sig = Lambda_22 - v_Lb' * v_Lb;   % conditional cov of f(tstar)
                                            % given f(t1) and f''(t2)

    % posterior of curvature given data at t1
    mean_dX = (L_K3' \ (L_K3 \ KK2))';      % A
    cov_dX  = cov_X - mean_dX * KK2;        % B
    L_dX    = jitterchol(cov_dX);

    % precompute terms reused in Gibbs
    L_sig = jitterchol(Lambda_sig);         % chol of Lambda_sig
    R     = jitterchol(K3);                 % chol of K3
    Q     = R' \ (R \ eye(set_num));        % precision of f''(t2)
    Q     = (Q + Q')/2;

    % Gibbs sampling for curvature process (f'' >= 0)
    M    = 501;
    Zd   = zeros(set_num, M);   % raw curvature samples
    Zd_p = zeros(set_num, M);   % truncated at 0

    for k = 2:M
        mu_m    = zeros(set_num,1);
        nu_m    = zeros(set_num,1);
        theta_m = zeros(set_num,1);
        delta_m = zeros(set_num,1);

        for i = 1:set_num
            % conditional Normal using precision Q instead of submatrix chol
            Qi_i    = Q(i,i);
            Qi_rest = Q(i,:);  Qi_rest(i) = [];

            if i > 1 && i < set_num
                temp_zp  = [Zd(1:i-1, k);   Zd(i+1:set_num, k-1)];
                temp_zpp = [Zd_p(1:i-1, k); Zd_p(i+1:set_num, k-1)];
            elseif i == 1
                temp_zp  = Zd(2:set_num, k-1);
                temp_zpp = Zd_p(2:set_num, k-1);
            else
                temp_zp  = Zd(1:set_num-1, k);
                temp_zpp = Zd_p(1:set_num-1, k);
            end

            mu_i    = -(Qi_rest / Qi_i) * temp_zp;
            nu_i    = 1 / Qi_i;
            nu_sqrt = sqrt(nu_i);

            mu_m(i) = mu_i;
            nu_m(i) = nu_i;

            % conditional posterior-like proposal
            temp_a = mean_dX(:, i);
            if set_num ~= 1
                if i == 1
                    temp_A = mean_dX(:, 2:set_num);
                elseif i == set_num
                    temp_A = mean_dX(:, 1:set_num-1);
                else
                    temp_A = [mean_dX(:, 1:i-1), mean_dX(:, i+1:set_num)];
                end
            end

            if set_num ~= 1
                v_dX     = L_dX \ temp_a;
                temp_th1 = (x - mu) - temp_A * temp_zpp;
                alpha_dX = L_dX' \ (L_dX \ temp_th1);

                temp_th  = temp_a' * alpha_dX + mu_m(i) * (1/nu_m(i));
                temp_dt1 = (1/nu_m(i)) + (v_dX' * v_dX);
                temp_dt  = 1 / temp_dt1;

                theta_m(i) = temp_dt * temp_th;
                delta_m(i) = sqrt(temp_dt);
            else
                v_dX     = L_dX \ mean_dX;
                alpha_dX = L_dX' \ (L_dX \ (x - mu));
                temp_th  = mean_dX' * alpha_dX + mu_m(i) * (1/nu_m(i));
                temp_dt1 = (1/nu_m(i)) + (v_dX' * v_dX);
                temp_dt  = 1 / temp_dt1;

                theta_m(i) = temp_dt * temp_th;
                delta_m(i) = sqrt(temp_dt);
            end

            % mixture sampling enforcing f''(t2_i) >= 0
            eps      = 0;
            temp_ks  = normcdf(eps, mu_m(i), nu_sqrt);              % mass below 0
            temp_qx  = 1 - normcdf(eps, theta_m(i), delta_m(i));    % mass above 0

            temp_const = temp_qx * delta_m(i)/nu_sqrt * ...
                        exp( -mu_m(i)^2/(2*nu_m(i)) + theta_m(i)^2/(2*temp_dt) );
            q_const  = temp_ks / (temp_ks + temp_const);
            if isinf(temp_const), q_const = 1; end
            if isnan(temp_const)
                warning('NaN value'); break;
            end
            if temp_ks==0 && temp_const==0
                temp_ks = 1e-6;
                q_const = temp_const/(temp_ks+temp_const);
            end

            biv_num = binornd(1, q_const);
            if biv_num == 0
                u         = temp_ks * rand(1);
                Zd(i,k)   = norminv(u, mu_m(i), nu_sqrt);
                Zd_p(i,k) = 0;
            else
                u         = 1 - (temp_qx - temp_qx * rand(1));
                Zd(i,k)   = norminv(u, theta_m(i), delta_m(i));
                Zd_p(i,k) = Zd(i,k);
            end
        end
    end

    % update violation probabilities for remaining candidates
    if (nstar - set_num) > 0
        alpha_K3 = L_K3' \ (L_K3 \ mean(Zd_p(:,51:M), 2));
        MU       = mu_lam + Sigma_12 * alpha_K3;

        mu_z = MU(n+1 : n+lg_star);
        mu_x = MU(1:n);

        Z_mu    = mu_z + Lambda_21 * (L_Lb' \ (L_Lb \ (x - mu_x)));
        Z_sigma = Lambda_sig / (M - 50);

        prob_y  = normcdf( zeros(nstar - set_num,1), ...
                            Z_mu, ...
                            diag(sqrt(Z_sigma)) );
    else
        t2 = tstar_g;
        break;
    end
end

t2    = sort(t2);
m     = length(t2);
tstar = tstar_g;

% ============================================================
% 3) Final Gibbs sampling on the full grid (M = 10000)
% ============================================================
cov_X  = feval(covfunc{:}, loghyper, t1);

[K1, K2, K3, KK2] = covGP2([log(ell);log(sigma_f)], t1, t2);
L_covX     = jitterchol(cov_X);
alpha_covX = L_covX' \ (L_covX \ (x - mu));
L_K3       = jitterchol(K3);

mean_dX = (L_K3' \ (L_K3 \ KK2))';
cov_dX  = cov_X - mean_dX * KK2;
L_dX    = jitterchol(cov_dX);

[Ks1, Ks2, Ks3, KKs2] = covGP2([log(ell);log(sigma_f)], tstar, t2); %#ok<ASGLU>
cov_Zxt = covSEisoS([log(ell);log(sigma_f)], t1,    tstar);
cov_Zts = covSEisoS([log(ell);log(sigma_f)], tstar);

Sigma_11 = [cov_X,  cov_Zxt;
            cov_Zxt', cov_Zts];
Sigma_12 = [K2;
            Ks2];
Sigma_21 = [KK2, KKs2];
Sigma_22 = K3;

v_K3     = L_K3 \ Sigma_21;
Lambda   = Sigma_11 - v_K3' * v_K3;

Lambda_11 = Lambda(1:n,           1:n);
Lambda_12 = Lambda(1:n,           n+1:n+nstar);
Lambda_21 = Lambda(n+1:n+nstar,   1:n);
Lambda_22 = Lambda(n+1:n+nstar,   n+1:n+nstar);

L_Lb     = jitterchol(Lambda_11);
v_Lb     = L_Lb \ Lambda_12;
Lambda_sig = Lambda_22 - v_Lb' * v_Lb;

L_sig = jitterchol(Lambda_sig);
R     = jitterchol(K3);
Q     = R' \ (R \ eye(m));
Q     = (Q + Q')/2;

M    = 10000;
Z    = zeros(nstar, M);
Zd   = zeros(m,     M);
Zd_p = zeros(m,     M);

mu_lam = mu * ones(n + nstar, 1);

for k = 2:M
    mu_m    = zeros(m,1);
    nu_m    = zeros(m,1);
    theta_m = zeros(m,1);
    delta_m = zeros(m,1);

    % Gibbs update of f''(t2)
    for i = 1:m
        Qi_i    = Q(i,i);
        Qi_rest = Q(i,:);  Qi_rest(i) = [];

        if i > 1 && i < m
            temp_zp  = [Zd(1:i-1, k);   Zd(i+1:m, k-1)];
            temp_zpp = [Zd_p(1:i-1, k); Zd_p(i+1:m, k-1)];
        elseif i == 1
            temp_zp  = Zd(2:m, k-1);
            temp_zpp = Zd_p(2:m, k-1);
        else
            temp_zp  = Zd(1:m-1, k);
            temp_zpp = Zd_p(1:m-1, k);
        end

        mu_i     = -(Qi_rest / Qi_i) * temp_zp;
        nu_i     =  1 / Qi_i;
        nu_sqrt  = sqrt(nu_i);

        mu_m(i) = mu_i;
        nu_m(i) = nu_i;

        temp_a = mean_dX(:, i);
        if i == 1
            temp_A = mean_dX(:, 2:m);
        elseif i == m
            temp_A = mean_dX(:, 1:m-1);
        else
            temp_A = [mean_dX(:, 1:i-1), mean_dX(:, i+1:m)];
        end

        v_dX     = L_dX \ temp_a;
        temp_th1 = (x - mu) - temp_A * temp_zpp;
        alpha_dX = L_dX' \ (L_dX \ temp_th1);

        temp_th  = temp_a' * alpha_dX + mu_m(i) * (1/nu_m(i));
        temp_dt1 = (1/nu_m(i)) + (v_dX' * v_dX);
        temp_dt  = 1 / temp_dt1;

        theta_m(i) = temp_dt * temp_th;
        delta_m(i) = sqrt(temp_dt);

        eps      = 0;
        temp_ks  = normcdf(eps, mu_m(i), nu_sqrt);
        temp_qx  = 1 - normcdf(eps, theta_m(i), delta_m(i));

        temp_const = temp_qx * sqrt(delta_m(i)/nu_sqrt) * ...
                    exp(-mu_m(i)^2/(2*nu_m(i)) + theta_m(i)^2/(2*temp_dt));

        q_const  = temp_const / (temp_ks + temp_const);
        if isinf(temp_const), q_const = 1; end
        if isnan(temp_const)
            warning('NaN value'); break;
        end
        if temp_ks==0 && temp_const==0
            temp_ks = 1e-6;
            q_const = temp_const/(temp_ks+temp_const);
        end

        biv_num = binornd(1, q_const);
        if biv_num == 1
            u         = temp_ks * rand(1);
            Zd(i,k)   = norminv(u, mu_m(i), nu_sqrt);
            Zd_p(i,k) = 0;
        else
            u         = 1 - (temp_qx - temp_qx*rand(1));
            Zd(i,k)   = norminv(u, theta_m(i), delta_m(i));
            Zd_p(i,k) = Zd(i,k);
        end
    end

    % conditional draw of f(tstar)
    alpha_K3 = L_K3' \ (L_K3 \ Zd_p(:,k));
    MU       = mu_lam + Sigma_12 * alpha_K3;

    mu_z = MU(n+1:n+nstar);
    mu_x = MU(1:n);

    Z_mu   = mu_z + Lambda_21 * (L_Lb' \ (L_Lb \ (x - mu_x)));
    Z(:,k) = Z_mu + L_sig * randn(nstar, 1);
end

% ============================================================
% 4) Summary statistics
% ============================================================
Zval = Z(:, 5001:10000);
Ef   = median(Zval, 2);
Varf = var(Zval, 0, 2);

ci    = quantile(Zval', [0.025, 0.975]);
ci_lo = ci(1,:)';
ci_hi = ci(2,:)';

epsv = 1e-12;

mse      = mean((Ef - y_true).^2);
nlpd     = -mean( log( max(realmin, normpdf(y_true, Ef, sqrt(max(Varf, epsv)))) ) );
coverage = mean( (y_true >= ci_lo) & (y_true <= ci_hi) );
width    = mean( ci_hi - ci_lo );

elapsedTime = toc(tAll);

% Visualization (1D posterior vs truth)
figure;
plot(tstar_in, Ef, '.b', 'MarkerSize', 17); hold on;
plot(tstar_in, y_true, '.r', 'MarkerSize', 17);
plot(tstar_in, ci_lo, '-.b', 'LineWidth', 1);
plot(tstar_in, ci_hi, '-.b', 'LineWidth', 1);
legend('Posterior Median','True Function','Lower 95% CI','Upper 95% CI', ...
    'Location','Best');
grid on; hold off;

end
