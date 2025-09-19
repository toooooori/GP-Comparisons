clear; clc;

% === 改这里：本地数据文件夹（含 test_grid.csv 和 train_runXX.csv）===
data_root = 'datasets_stepwise_csv';   % e.g., 'datasets_sigmoid_csv' / 'datasets_shrink_csv'

% 读取测试集（公共网格）
Ttest  = readtable(fullfile(data_root, 'test_grid.csv'));
xt     = Ttest.xt(:);
y_true = Ttest.y_true(:);

n_trials = 25;
results  = zeros(n_trials, 5);   % [MSE, NLPD, Coverage, Width, Time]

for i = 1:n_trials
    f1 = fullfile(data_root, sprintf('train_run%02d.csv', i));
    T = readtable(f1);
    x = T.x(:);
    y = T.y(:);

    [mse, nlpd, coverage, width, tsec] = IP(x, y, xt, y_true);
    results(i,:) = [mse, nlpd, coverage, width, tsec];
end
% 汇总
mean_results = mean(results); std_results = std(results);
fprintf('\nAverage over %d runs:\n', n_trials);
fprintf('MSE: %.4f ± %.4f\n', mean_results(1), std_results(1));
fprintf('NLPD: %.4f ± %.4f\n', mean_results(2), std_results(2));
fprintf('Coverage: %.4f ± %.4f\n', mean_results(3), std_results(3));
fprintf('Width: %.4f ± %.4f\n', mean_results(4), std_results(4));
fprintf('Time (sec): %.4f ± %.4f\n', mean_results(5), std_results(5));

save('results_mono_ip_stepwise.mat', 'results');
