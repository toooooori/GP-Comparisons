% DEMO_MONOTONIC_2D  Fit a 2D GP with monotonicity in x2 only
root = 'gpstuff';
run(fullfile(root, 'startup.m'));
%% 1. Generate training data
tic;
rng(0);
n   = 50;                % number of training points
X   = rand(n,2);         % uniform on [0,1]^2
x1  = X(:,1); x2 = X(:,2);
% true function + noise
f_true = 1./(1+exp(-10*(x1-0.4))) + 1./(1+exp(-10*(x2-0.6)));
%f_true = exp(x1) + x2 + sin(20*x2 - 10)./20;
sigma   = 0.5;
y_obs   = f_true + sigma*randn(n,1);

gpcf=gpcf_sexp();
lik=lik_gaussian();

gp=gp_set('cf', gpcf, 'lik', lik, 'jitterSigma2', 1e-6);

% 3. Enforce monotonicity only in the second input dimension (x2)
%nvd = [1 2];
nvd = [2];  % dimension index to constrain
opt = optimset('TolX',1e-4,'TolFun',1e-4,'Display','off','MaxIter',100);
gp = gp_monotonic(gp, X, y_obs, ...
                  'nvd',      nvd, ...
                  'nv',       20, ...       % initial virtual points
                  'optimize', 'on', ...
                  'opt',      opt, ...
                  'optimf',   @fminscg);

%% 4. Do CCD inference and prediction on a grid
gpia = gp_ia(gp, X, y_obs);

ngrid = 100;
[x1g,x2g] = meshgrid(linspace(0,1,ngrid), linspace(0,1,ngrid));
Xgrid     = [x1g(:), x2g(:)];
[Ef_mono, Varf_mono] = gp_pred(gpia, X, y_obs, Xgrid);


%% 5. Visualization
figure;
% posterior mean surface
Zmean = reshape(Ef_mono, ngrid, ngrid);
mesh(x1g, x2g, Zmean, 'FaceColor', 'b', 'EdgeColor', 'b');
hold on;
% true function surface outline (no noise)
Ztrue = reshape(1./(1+exp(-10*(x1g-0.4))) + 1./(1+exp(-10*(x2g-0.6))), ngrid, ngrid);
mesh(x1g, x2g, Ztrue, 'FaceColor', 'r', 'EdgeColor', 'r');
legend('Predicted mean', 'True function', 'Location', 'Best');
hold off;

%% 6. Compute and display MSE on grid
y_pred_vec = Ef_mono;               % nstar×1
y_true_vec = Ztrue(:);              % nstar×1
MSE_grid   = mean((y_pred_vec - y_true_vec).^2);
fprintf('Grid MSE = %.6f\n', MSE_grid);

elapsedTime = toc;
fprintf('Elapsed time: %.2f sec\n', elapsedTime);

SSres = sum((y_true_vec - y_pred_vec).^2);
SStot = sum((y_true_vec - mean(y_true_vec)).^2);
R2    = 1 - SSres/SStot;
fprintf('R^2 = %.3f\n', R2);
