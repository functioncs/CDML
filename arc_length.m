function value = arc_length(M, A, B)
% value = arc_length(M, A, B)
% Input: M(j, :) denotes the j-dimensional polyno. coefficients, 
%         y_j =  M(j, 1)*x + ... +M(j, c)*x^c 
% Output: The arc_length value of T from A to B

    % The number of intervals
    L = 1000;
    
    % Calculate arc length by interval approximation
    [d, c] = size(M);
    a = min(A, B); b = max(A, B);
    delta_t = (b - a)/L;
    inter_values = a + (0:L-1)*delta_t;
    
    % Pre-calcualte power values at intervals
    power_inter_values = repmat(inter_values, c, 1) ...
          .^ repmat((0:c-1)', 1, length(inter_values));
    
    % Convert poly. computations to linear projections
    % M(:, 1) are first-order coefficients 
    derivate_coefficients_with_const = M .* repmat(1:c, d, 1);
    % T_derivate_coefficients(:, 1) are zeror-order coefficients
    result = derivate_coefficients_with_const * power_inter_values;
    value = sum(sqrt(sum(result .* result, 1))) * delta_t;
end