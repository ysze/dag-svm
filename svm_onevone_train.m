function [alpha, bias, X_new, Y_new] = svm_onevone_train(X, Y, class_label_1, class_label_2, degree)
% One-vs-One SVM

% Extract only the two target classes
ind = find(Y==class_label_1 | Y==class_label_2);
Y = Y(ind);
X = X(ind, :);
Y = 2 * (Y==class_label_1) - 1;
X_new = X;
Y_new = Y;

% Get number of training samples
n = length(Y);

% Define function to be minimized
H = poly_kernel(X, X, degree) .* (Y * Y');
f = -ones(n, 1);
zero_vec = zeros(n, 1);

% Define constraints
A = -eye(n);
a = zeros(n, 1);
B = Y';
b = 0;

% Define penalty parameter
c = ones(n, 1) * 1e1;

% Solve for alpha's
threshold = 1e-4;
alpha = quadprog(H, f, A, a, B, b, zero_vec, c);
alpha = alpha .* (alpha > threshold);

% Get support vectors
sv_ind = alpha~=0;
Y_sv = Y(sv_ind);
alpha_sv = alpha(sv_ind);
X_sv = X(sv_ind);

% Compute bias
bias = mean(Y_sv' - ((alpha_sv .* Y_sv)' * poly_kernel(X_sv, X_sv, degree)));

end
