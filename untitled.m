T = readmatrix('sr_sigmoid.csv');
x = T(:,1); mu = T(:,5); lo = T(:,3); up = T(:,4); y_true = T(:,2);

figure; hold on
plot(x, up, '-.b','LineWidth', 1, 'MarkerSize', 17);
plot(x, lo, '-.b','LineWidth', 1, 'MarkerSize', 17);
plot(x, mu, '.b','MarkerSize', 17);
plot(x, y_true, '.r','MarkerSize', 17);
legend('Posterior Mean','Real Values','Lower 95% Credible Band','Upper 95% Credible Band','Location','Best');
hold off;
