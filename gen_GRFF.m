function [grff, grff_test, timer] = gen_GRFF(X, X_test, pos_cdf, neg_cdf, kernelpower_pos_coeff, kernelpower_neg_coeff, w_total, w, dim, mapdim)
        tic
        W_pos = interp1(pos_cdf, w_total, random('unif',0,1,mapdim,1), 'linear', 0);
        W_pos = W_pos';
        w_pos = normrnd(0,1,dim,mapdim);
%         
        w_pos = bsxfun(@times,1./sqrt(sum(w_pos.^2)),w_pos);        
        W_pos = repmat(W_pos, dim, 1).*w_pos;
        rff_x_pos = [sqrt(kernelpower_pos_coeff)*cos((X*W_pos)), sqrt(kernelpower_pos_coeff)*sin((X*W_pos))];
        rff_x_pos_test = [sqrt(kernelpower_pos_coeff)*cos((X_test*W_pos)), sqrt(kernelpower_pos_coeff)*sin((X_test*W_pos))];
        
        
        W_neg = interp1(neg_cdf, w, random('unif',0,1,mapdim,1), 'linear', 0);
        W_neg = W_neg';
        w_neg = normrnd(0,1,dim,mapdim);
        
        w_neg = bsxfun(@times,1./sqrt(sum(w_neg.^2)),w_neg);
        W_neg = repmat(W_neg, dim, 1) .* w_neg;

        rff_x_neg = [(1i)*sqrt(kernelpower_neg_coeff)*cos((X*W_neg)), (1i)*sqrt(kernelpower_neg_coeff)*sin((X*W_neg))];
        rff_x_neg_test = [(1i)*sqrt(kernelpower_neg_coeff)*cos((X_test*W_neg)), (1i)*sqrt(kernelpower_neg_coeff)*sin((X_test*W_neg))];
        
        grff = sqrt(1/mapdim)*[rff_x_pos, rff_x_neg];
        grff = grff.';
        
        
        grff_test = sqrt(1/mapdim)*[rff_x_pos_test, rff_x_neg_test];
        grff_test = grff_test.';
        timer = toc;
        timer = timer;
end