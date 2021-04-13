function [ rff_x, rff_x_test, time ] = gen_RFF(DATA, DATA_TEST, a, b, num_gaussians, CS_COL)
% Wrapper function to generate the RM

[N, dim] = size(DATA);
[N_test,dim_test] = size(DATA_TEST);

params.dim = dim;
kernelparams.a = a;
kernelparams.b = b;
cdf = [];
model_file = sprintf('model_a%d_b%d_dim%d_num_gaussians%d.mat', kernelparams.a, kernelparams.b, params.dim, num_gaussians);
if (~exist(model_file, 'file'))
    error('Model file %s does not exisit. Run offline_model_generation to generate the file first.', model_file);
end
load (model_file);

% figure
% plot(ww,cdf)
tic
W = interp1(cdf, ww, rand(CS_COL,1), 'linear', 0);
W = W';
w = normrnd(0,1,params.dim,CS_COL);
w = bsxfun(@times,1./sqrt(sum(w.^2)),w);
b = rand(CS_COL,1)*2*pi;

b_train = repmat(b', N, 1);
b_test = repmat(b',N_test,1);
W = repmat(W, size(w, 1), 1) .* w;
rff_x = cos((DATA * W  + b_train)) * sqrt(2 / CS_COL);
rff_x_test = cos((DATA_TEST * W  + b_test)) * sqrt(2 / CS_COL);
rff_x = rff_x.';
rff_x_test = rff_x_test.';
time = toc;

end