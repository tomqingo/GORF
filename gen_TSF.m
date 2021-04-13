%% kernel TS technique
function [tsf_train, tsf_test, time_tsf] = gen_TSF(X_train_trans, X_test_trans, mapdim, dim)
       N_train = length(X_train_trans);
       N_test = length(X_test_trans);
       R = size(X_train_trans,2);
       
       h = zeros(mapdim,1);
       s = zeros(dim+1,1);
       tsf_train = zeros(N_train,mapdim);
       tsf_test = zeros(N_test, mapdim);
       
       tic
    
       % construct hash table
       for jj = 1:(dim+1)
           h(jj) = randi(mapdim);
       end
    
       for jj = 1:(dim+1)
           s(jj) = 2*round(rand(1))-1;
       end
    
       % count sketch in Matlab
       for jj = 1:mapdim
           index = find(h == jj);
           if isempty(index)
              continue;
           else
               for kk = 1:length(index)
                   tsf_train(:,jj) = tsf_train(:,jj) + s(index(kk))*X_train_trans(:,index(kk));
                   tsf_test(:,jj) = tsf_test(:,jj) + s(index(kk))*X_test_trans(:,index(kk));
               end
           end
       end
       
       tsf_train = tsf_train.';
       tsf_test = tsf_test.';
       
       time_tsf = toc;
end