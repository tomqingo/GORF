function [gorf, gorf_test, timer] = gen_GORF_mix(X, X_test, pos_cdf, neg_cdf, kernelpower_pos_coeff, kernelpower_neg_coeff, w_total, w, dim, mapdim)
        tic
        W_pos = interp1(pos_cdf, w_total,random('unif',0,1,mapdim,1), 'linear', 0);
        W_pos = W_pos';
        if dim<mapdim
            w_col = normrnd(0,1,mapdim*2,mapdim*2);
            [w_col, R_0] = qr(w_col);
            w_pos = w_col(1:dim, 1:mapdim);
            w_neg = w_col(1:dim, mapdim+1:end);
        else
            w_col = normrnd(0,1,dim*2,dim*2);
            [w_col, R_0] = qr(w_col);
            w_pos = w_col(1:dim, 1:mapdim);
            w_neg = w_col(1:dim, mapdim+1:2*mapdim);
        end
        w_pos = bsxfun(@times, 1./sqrt(sum(w_pos.^2)), w_pos);        
        W_pos = repmat(W_pos, dim, 1).*w_pos;
        rff_x_pos = [sqrt(kernelpower_pos_coeff)*cos((X*W_pos)), sqrt(kernelpower_pos_coeff)*sin((X*W_pos))];
        rff_x_pos_test = [sqrt(kernelpower_pos_coeff)*cos((X_test*W_pos)), sqrt(kernelpower_pos_coeff)*sin((X_test*W_pos))];
            
        W_neg = interp1(neg_cdf, w, random('unif',0,1,mapdim,1), 'linear', 0);
        W_neg = W_neg';
        w_neg = bsxfun(@times, 1./sqrt(sum(w_neg.^2)), w_neg);
        W_neg = repmat(W_neg, dim, 1) .* w_neg;
        rff_x_neg = [(1i)*sqrt(kernelpower_neg_coeff)*cos((X*W_neg)), (1i)*sqrt(kernelpower_neg_coeff)*sin((X*W_neg))];
        rff_x_neg_test = [(1i)*sqrt(kernelpower_neg_coeff)*cos((X_test*W_neg)), (1i)*sqrt(kernelpower_neg_coeff)*sin((X_test*W_neg))];
        
        gorf = sqrt(1/mapdim)*[rff_x_pos,rff_x_neg];
        gorf_test = sqrt(1/mapdim)*[rff_x_pos_test, rff_x_neg_test];
        gorf = gorf.';
        gorf_test = gorf_test.';
        timer = toc;
end