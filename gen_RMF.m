%% kernel RM technique
function [rmf_train, rmf_test, time_rmf] = gen_RMF(X_train, X_test, alpha, q, p, poss, mapdim, probability_cdf, n)
    num_train = size(X_train,1);
    num_test = size(X_test,1);
    dim = size(X_train,2);
    rmf_train = zeros(num_train,mapdim);
    rmf_test = zeros(num_test,mapdim);
    tic
    
    for jj = 1:mapdim
        rand_num = rand(1);
        [value,index] = min(abs(probability_cdf-rand_num));
        index_min = min(index);
        n_select = n(index_min);
        if n_select == 0
           a = alpha*(q).^(p);
           rmf_train(:,jj) = sqrt(a*poss)*ones(num_train,1);
           rmf_test(:,jj) = sqrt(a*poss)*ones(num_test,1);
        else
           w = rand(n_select,dim);
           w = round(w);
           w = 2*w - 1;
           product_matrix_train = w*X_train';
           product_matrix_test = w*X_test';
        if n_select == 1
           a = alpha;
        else
           a = 0;
        end   
        rmf_train(:,jj) = sqrt(a*poss^2)*prod(product_matrix_train,1)';
        rmf_test(:,jj) = sqrt(a*poss^2)*prod(product_matrix_test,1)';
        end
    end
    
    rmf_train = 1/sqrt(mapdim)*rmf_train;
    rmf_test = 1/sqrt(mapdim)*rmf_test;
    
    rmf_train = rmf_train.';
    rmf_test = rmf_test.';
    time_rmf = toc;
end