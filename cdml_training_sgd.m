function tensor_M = cdml_training_sgd(X, X_hat, labels, parameters)
% tensor_M = cdml_training_sgd(X, X_hat, labels, lambda, c, m, batch_size, eta)
% Input: X and X_hat are d by N; labels is N-dimensional; 
%        lambda is regularization parameter, eta is learning rate.
% Output: tensor_M are (d by c by m)
    lambda = parameters.lambda;
    c = parameters.c;
    m = parameters.m;
    batch_size = parameters.batch_size;
    eta = parameters.eta;
    tensor_M = parameters.tensor_M;
    epoch = parameters.epoch;
    
    [d, N] = size(X);
    obj_value = obj_empirical_L(tensor_M, X, X_hat, labels) ...
                          + lambda * obj_regularizer_R(tensor_M);
    fprintf('Init: loss value = %f\n', obj_value);
    for i = 1 : epoch
        indexs = randperm(N);
        batch_gap_check = 3;
        batched = 0;
        for j = 1: batch_size: N
            batch_indexs = indexs(j:min(j + batch_size - 1, N));
            batch_num = length(batch_indexs);
            [batch_grads_M, batch_squared_dists] ...
            = batch_gradient_squared_dist(tensor_M, X(:, batch_indexs), ...
            X_hat(:, batch_indexs));
            batch_grads_squared_dist = batch_gradient_empirical_squared_dist(batch_squared_dists, labels(batch_indexs));
            % batch_grads_M is d by c by m by $batch_num$
            % batch_grads_squared_dist is $batch_num$-dimensional
            temp = repmat(reshape(batch_grads_squared_dist, 1, 1, 1, batch_num), d, c, m, 1);
            grad_list = batch_grads_M .* temp;
            grad = sum(grad_list, 4);
            tensor_M = tensor_M - eta*(grad + lambda*gradient_regularizer(tensor_M));
            batched  = batched + 1;
            if mod(batched, batch_gap_check)==0
                obj_value = obj_empirical_L(tensor_M, X, X_hat, labels) ...
                          + lambda * obj_regularizer_R(tensor_M);
                fprintf('epoch_%d, batch_%d: loss value = %f\n', ...
                    i, batched, obj_value);
            end
        end
    end
end

function empirical_loss_value = obj_empirical_L(tensor_M, X, X_hat, labels)
% empirical_loss_value = obj_empirical_L(tensor_M, X, X_hat, labels)
% Input: tensor_M (d by c by m); X and X_hat are d by N;
%        labels is N-dimensional with 0-1 values
    [d, c, m] = size(tensor_M);
    [d, N] = size(X);
    empirical_loss_value = 0;
    u = 2; v = 8; % contrastive
    for i= 1 : N
        empirical_loss_value = empirical_loss_value ...
            + labels(i)*max(0, squared_distance_value(tensor_M, X(:, i), X_hat(:, i))-u)^2 ...
            + (1- labels(i))*max(0, v - squared_distance_value(tensor_M, X(:, i), X_hat(:, i)))^2;
    end
    empirical_loss_value = empirical_loss_value / N;
end

function grad = batch_gradient_empirical_squared_dist(squared_distance_values, labels)
% grad = batch_gradient_empirical_squared_dist(distance_values)
% Calculate the gradient of empirical loss w.r.t. squared_distance
% Input: distance_values and labels are N-dimensional
% Output: grad is N-dimensional
    u = 2; v = 8; % contrastive
    grad = labels .* (2 * ((squared_distance_values>u).* (squared_distance_values - u)))...
         - (1 - labels) .* (2 * ((squared_distance_values<v).* (v - squared_distance_values)));
end

function value = obj_regularizer_R(tensor_M)
% value = obj_regularizer_R(tensor_M).
% F-norm.
    result = tensor_M .* tensor_M;
    value = sum(result(:));
end

function grad = gradient_regularizer(tensor_M)
    grad = 2 * tensor_M;
end



