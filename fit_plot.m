%T = readmatrix('bf_parabola.csv');
%x = T(:,1); mu = T(:,5); lo = T(:,3); up = T(:,4); y_true = T(:,2);

%figure; hold on
%plot(x, up, '-.b','LineWidth', 1, 'MarkerSize', 17);
%plot(x, lo, '-.b','LineWidth', 1, 'MarkerSize', 17);
%plot(x, mu, '.b','MarkerSize', 17);
%plot(x, y_true, '.r','MarkerSize', 17);
%legend('Posterior Mean','Real Values','Lower 95% Credible Band','Upper 95% Credible Band','Location','Best');
%grid on;hold off;

T = readtable('bf_2D_results.csv');
x = T.x; y = T.y;
xu = unique(x); yu = unique(y);
[X, Y] = meshgrid(xu, yu);
Real = reshape(T.z_true,      [numel(xu), numel(yu)])';
Mu = reshape(T.z_post_mean, [numel(xu), numel(yu)])';

figure;
mesh(yt1, yt2, Mu, 'FaceColor', 'b', 'EdgeColor', 'b');
hold on;
mesh(yt1, yt2, Real, 'FaceColor', 'r', 'EdgeColor', 'r');
hold off;
legend('Predicted mean', 'True function', 'Location', 'Best');