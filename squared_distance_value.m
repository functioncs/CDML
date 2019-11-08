function value = squared_distance_value(tensor_M, x, x_hat)
% value = squared_distance_value(tensor_M, x, x_hat)
% Input: tensor_M (d by c by m); x and x_hat are d-dimensional
% Output: squared distance value
    [d, c, m] = size(tensor_M);
    value = 0;
    for i = 1 : m
        value = value ...
            + arc_length(tensor_M(:,:,i), 0, 1)^2 ...
            * arc_length(tensor_M(:,:,i), solve_ft(tensor_M(:,:,i), x), ...
            solve_ft(tensor_M(:,:,i),  x_hat))^2;
    end
%     arc1 = arc_length(tensor_M(:,:,i), 0, 1);
%     a = solve_ft(tensor_M(:,:,i), x);
%     b = solve_ft(tensor_M(:,:,i),  x_hat);
%     arc2 = arc_length(tensor_M(:,:,i), a, b);
%     fprintf('Check distance_value %f\n', value);
end

