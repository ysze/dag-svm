function pred = svm_dag(X_train, Y_train, X_test, degree)
% DAG SVM implementation

% Get all possible classes
classes = unique(Y_train);

% Get number of classes
n_classes = length(classes);

% Initialize cells to store training results
alpha_cell = cell(n_classes);
bias_cell = cell(n_classes);
X_new_cell = cell(n_classes);
Y_new_cell = cell(n_classes);

% Initialize variable to store predictions
pred = zeros(size(X_test, 1), 1);

% Train classifier
for i = 1:n_classes
    for j = i+1:n_classes
        [alpha, bias, X_new, Y_new] = svm_onevone_train(X_train, Y_train, i-1, j-1, degree);
        alpha_cell{i, j} = alpha;
        bias_cell{i, j} = bias;
        X_new_cell{i, j} = X_new;
        Y_new_cell{i, j} = Y_new;
    end
end

% Predict on the test data
for i = 1:size(X_test, 1)
    remain = classes;
    while length(remain) > 1
        alpha = alpha_cell{remain(1) + 1, remain(end) + 1};
        bias = bias_cell{remain(1) + 1, remain(end) + 1};
        X_new = X_new_cell{remain(1) + 1, remain(end) + 1};
        Y_new = Y_new_cell{remain(1) + 1, remain(end) + 1};
        out = (alpha .* Y_new)' * poly_kernel(X_new, X_test(i, :), degree) + bias;
        if out > 0
            remain = remain(1:end-1);
        else
            remain = remain(2:end);
        end
    end
    pred(i) = remain;
end

end
