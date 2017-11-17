function k = poly_kernel(X_m, X_n, degree)
% Polynomial kernel

k = (X_m * X_n' + 1) .^ degree;
end