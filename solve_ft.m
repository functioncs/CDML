function [minimizer_root, minimum_value] = solve_ft(M, x)
% [minimizer_root] = solve_ft(M, x)
% [minimizer_root, minimum_value] = solve_ft(M, x)
% Find out the calibration (nearest) point.
% Input: For the given M (d x c), and x (d-dimensional), 
%        solve min_t f(t) = ||M(t) - x||_2^2, 
%        where M_j(t) = M(j,1)*t^1+...M(j,c)*t^c.
% Output: The minimizer t^star.

    [~, c] = size(M);
    coefficients = zeros(2*c, 1); % From 1-order to 2(c-1)-order

    % Caculate result_1(j, k) = M(:, j)'*M(:, k) for t^(j+k)
    result_1 = M' * M;
    % Caculate result_2(k) = M(:, k)'*x for t^k
    result_2 = -2 * M' * x;
    % Combine the final coefficients
    index_sum = repmat(1:c, c, 1) + repmat((1:c)', 1, c);
    for i = 2 : 2*c
         coefficients(i) = coefficients(i) + ...
             sum(sum(result_1(index_sum == i)));
    end
    coefficients = coefficients + [result_2; zeros(c, 1)];
    
    % Obtain the derivate coefficients and the corresponding roots
    derivate_coefficients_with_const = coefficients .* (1: 2*c)';
    all_roots = roots(derivate_coefficients_with_const(end:-1:1));
    
    % Test obj. values of all real roots
    real_roots = all_roots(is_real_root(all_roots));
    real_roots_objs = Mtx(real_roots);
    % fprintf('Check the M(0) = %f,\n',Mtx(0));
    min_value = min(real_roots_objs);
    
    % There exists multiple minminizers.
    is_min = (real_roots_objs - min_value == 0);
    minimizer_roots = real_roots(is_min);
    
    % Use the value-smallest minimizer as the calibration point.
    minimizer_root = min(minimizer_roots);
    if nargout >= 2
        minimum_value = min_value;
    end
    
    function index_logic = is_real_root(all_roots)
    % User-defined rule for filtering real number.
        tor = 1e-7;
        imag_num = imag(all_roots);
        index_logic = abs(imag_num) < tor;
    end

    function values = Mtx(ts)
    % Input: Here ts is a column vector.
    % Output: All obj. values for ts.
        [num, ~] = size(ts);
        if num == 0
            error('Test root is none.')
        end
        power_values = repmat(ts, 1, c) .^ repmat(1:c, num, 1);
        Mts = M * power_values';
        results = Mts - repmat(x, 1, num);
        values = sum(results.*results, 1);
    end
end