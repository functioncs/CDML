function [grads, squared_dists] = batch_gradient_squared_dist(tensor_M, X, X_hat)
% [grads, squared_dists] = batch_gradient_squared_dist(tensor_M, X, X_hat)
% Caculate the batch gradients and squared distances.
% Input: tensor_M (d by c by m), X,X_hat (d by batch_num)
% Output: grads (d by c by m by batch_num);
%         squared_dists (N-dimensional)
% Here d, c, m are data dimension, poly. order, measurer counts.
    [d, c, m] = size(tensor_M);
    [d, batch_num] = size(X);
    grads = zeros(d, c, m, batch_num);
    squared_dists = zeros(batch_num, 1);
    parfor i = 1 : batch_num
        for j = 1 : m
            Mj = tensor_M(:, :, j);
            x = X(:, i);
            x_hat = X_hat(:, i);
            cali_x = solve_ft(Mj, x);
            cali_x_hat = solve_ft(Mj, x_hat);
            length_unit = arc_length(Mj, 0, 1);
            length_cali = arc_length(Mj, cali_x, cali_x_hat);
            grads(:, :, j, i)...
            = 2*length_unit * gradient_arc_length(Mj, 0, 1)* length_cali^2 ...
            + length_unit^2 * gradient_cali_squared_arc_length(Mj, x, x_hat);
            %fprintf('one example, one measurer line.\n');
            squared_dists(i) = squared_dists(i) + length_unit^2 * length_cali^2;
        end
    end
end

function grad = gradient_arc_length(Mi, A, B)
% grad = gradient_arc_length(Mi, A, B)
% Input: Mi (d x c)
% Output: grad (d x c)
    
    [d, c] = size(Mi);
    L = 1000;
    a = min(A, B); b = max(A, B);
    delta_t = (b - a)/L;
    inter_values = a + (0:L-1)*delta_t;
    
    % Pre-calcualte power values at intervals
    power_inter_values = repmat(inter_values, c, 1) ...
        .^ repmat((0:c-1)', 1, length(inter_values)); % c x (L+1)
    % Derivate
    derivate_coefficients_with_const = Mi .* repmat(1:c, d, 1);
    result = derivate_coefficients_with_const * power_inter_values(1:c, :); % d x L
    result_1 = 1 ./ sqrt(sum(result .* result, 1)); % L x 1
    % Sum operation
    grad = delta_t * (result * diag(result_1) * power_inter_values');

end

function grad = gradient_cali_squared_arc_length(Mi, x, x_hat)
% grad = gradient_cali_squared_arc_length(Mi, x, x_hat)
% Input: Mi (d x c); x, x_hat are d-dimensional.
% Output: grad (d x c).
    [d, c] = size(Mi);
    mu = 1e-1;
    delta_M = randn(d, c);
    mu_delta_M = mu * delta_M;
    norm_mu_delta = norm(mu_delta_M(:));
    Mi_move = Mi + mu_delta_M;
    
    cali_x = solve_ft(Mi, x);
    cali_x_hat = solve_ft(Mi, x_hat);
    cali_x_move = solve_ft(Mi_move, x);
    cali_x_hat_move = solve_ft(Mi_move, x_hat);
    
    grad = (sign(delta_M)/norm_mu_delta) .* (arc_length(Mi_move, cali_x_move, cali_x_hat_move)^2 ... 
                               - arc_length(Mi, cali_x, cali_x_hat)^2);
end

