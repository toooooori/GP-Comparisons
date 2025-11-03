function [mse, nlpd, coverage, width, time_elapsed] = IP(x, y, xt, y_true, nvd_arg)
% IP  Monotonic GP with standardization + larger jitter (no priors)
%     Robust two-stage call to gp_monotonic to avoid EP instability.
%
% Inputs:
%   x      [N x D] train inputs
%   y      [N x 1] train targets
%   xt     [M x D] test inputs
%   y_true [M x 1] test targets (for metrics)
%   nvd_arg monotonic in first (and only) input dim, 1 for increasing, -1
%   for decreasing, [1 1] for both increasing in 2D
%
% Outputs:
%   mse, nlpd, coverage, width, time_elapsed

    tic;
    setrandstream(0);

    %% === 0) Standardize x (by column) and y ===
    mx = mean(x,1);
    sx = std(x,0,1);    sx(sx==0) = 1;
    xz  = (x  - mx)./sx;
    xtz = (xt - mx)./sx;

    my = mean(y);
    sy = std(y);        if sy==0, sy = 1; end
    yz      = (y - my)/sy;
    y_truez = (y_true - my)/sy;

    %% === 1) Base GP: increase jitter (no priors) ===
    gpcf = gpcf_sexp();
    lik  = lik_gaussian();
    gp   = gp_set('cf', gpcf, 'lik', lik, 'jitterSigma2', 1e-5);  % ↑ jitter for stability
    if isfield(gp,'deriv'), gp = rmfield(gp,'deriv'); end         % safety

    opt = optimset('TolX',1e-4,'TolFun',1e-4,'Display','iter');

    %% === 2) Monotonic GP (two-stage: first no optimization, then optimize) ===
    % Soft constraints to avoid EP oscillations on stepwise/plateau regions
    nv0 = max(1, floor(size(x,1)/8));  % fewer virtual points initially
    nu0 = 1e-2;                        % larger nu = softer monotonic constraint
    %nvd_arg = 1;                       % monotonic in first (and only) input dim

    % Stage 1: EP without hyperparameter optimization
    gp1 = gp_monotonic(gp, xz, yz, ...
        'nvd', nvd_arg, 'nu', nu0, 'nv', nv0, 'init','sample', ...
        'force', false, 'optimize','off', 'opt', opt, 'optimf', @fminscg);

    % Stage 2: attempt optimization; fall back to stage-1 model if unstable
    try
        gp2 = gp_monotonic(gp1, xz, yz, ...
            'nvd', nvd_arg, 'nu', nu0, 'nv', nv0, 'init','sample', ...
            'force', false, 'optimize','on',  'opt', opt, 'optimf', @fminscg);
        gp_use = gp2;
    catch
        warning('Optimization unstable; falling back to non-optimized monotonic GP.');
        gp_use = gp1;
    end

    %% === 3) Prediction in standardized space ===
    [Eftz, Varftz] = gp_pred(gp_use, xz, yz, xtz, 'yt', y_truez);
    Varftz = max(Varftz, 1e-12);  % numerical floor

    % De-standardize to original scale
    Eft   = Eftz*sy + my;
    Varft = Varftz*(sy^2);

    %% === 4) Metrics on original scale ===
    mse = mean( (Eft(:) - y_true(:)).^2 );
    nlpd = -mean( log(normpdf(y_true(:), Eft(:), sqrt(Varft(:)))), 'omitnan' );
    ci_lo = Eft - 1.96*sqrt(Varft);
    ci_hi = Eft + 1.96*sqrt(Varft);
    coverage = mean( (y_true >= ci_lo) & (y_true <= ci_hi) );
    width = mean( ci_hi - ci_lo );

    time_elapsed = toc;

    %% === 5) Plot on original scale ===
    figure;
    plot(xt, Eft, '.b', 'MarkerSize', 17); hold on;
    plot(xt, y_true, '.r', 'MarkerSize', 17);
    plot(xt, ci_lo, '-.b', 'LineWidth', 1);
    plot(xt, ci_hi, '-.b', 'LineWidth', 1);
    legend('Posterior Mean','Real Values','Lower 95% Credible Band','Upper 95% Credible Band','Location','Best');
    grid on; hold off;

end
