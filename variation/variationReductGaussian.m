clear all
clc

%% delta-gaussian kernel difference
sigma_fir = 1;
sigma_sec = 10;
params.dim = 100;
compare_scale = 0.1;
s = params.dim;

z = 0:0.001:2;
ww = 10000;
num_points_w = 10000;
w = 0:ww/(2*num_points_w-1):ww;
dw = ww/(2*num_points_w-1);

kernelfunc = @(normz) (exp(-(normz.^2)/(2*sigma_fir^2))-exp(-(normz.^2)/(2*sigma_sec^2)));
a = kernelfunc(z);

kernelpower_pos = sigma_fir^(params.dim)*exp(-sigma_fir^2*w.^2/2)/((2*pi)^(params.dim/2));
kernelpower_neg = sigma_sec^(params.dim)*exp(-sigma_sec^2*w.^2/2)/((2*pi)^(params.dim/2));

kernel_pos = zeros(1,length(z));
kernel_neg = zeros(1,length(z));
kernel_pos_sec = zeros(1,length(z));
kernel_neg_sec = zeros(1,length(z));
for ii = 1:length(z)
    kernel_pos(ii)= sum(kernelpower_pos.*cos(w*z(ii)))*dw;
    kernel_neg(ii)= sum(kernelpower_neg.*cos(w*z(ii)))*dw;
    kernel_pos_sec(ii)= sum(kernelpower_pos.*cos(w*(2*z(ii))))*dw;
    kernel_neg_sec(ii)= sum(kernelpower_neg.*cos(w*(2*z(ii))))*dw;
end


%% calculate the cumulative distribution
pos_cdf = cumsum(kernelpower_pos./sum(kernelpower_pos));
neg_cdf = cumsum(kernelpower_neg./sum(kernelpower_neg));
[pos_cdf, pos_uniq_ind] = unique(pos_cdf);
[neg_cdf, neg_uniq_ind] = unique(neg_cdf);
w_total = w;
w_total = w_total(pos_uniq_ind);
w = w(neg_uniq_ind);

kernelpower_pos_coeff = 1;
kernelpower_neg_coeff = 1;
kernel = kernelpower_pos_coeff*kernel_pos-kernelpower_neg_coeff*kernel_neg;


%% sample from the corresponding individual rbf spectral components respectively
sample_num = 20000;
alpha = round(params.dim/2-1);
mean_pos_first = zeros(1,length(z));
mean_pos_sec = zeros(1,length(z));
mean_neg_first = zeros(1,length(z));
mean_neg_sec = zeros(1,length(z));

for ii = 1:length(z)
    W_pos_norm_col = interp1(pos_cdf, w_total, random('unif',0,1,2*sample_num,1), 'linear', 0);
    W_neg_norm_col = interp1(neg_cdf, w, random('unif',0,1,2*sample_num,1), 'linear', 0);
    W_pos_norm = sqrt(W_pos_norm_col(1:sample_num).^2+W_pos_norm_col(sample_num+1:end).^2);
    W_neg_norm = sqrt(W_neg_norm_col(1:sample_num).^2+W_neg_norm_col(sample_num+1:end).^2);
    
    pos_first_inside_div = besselj(alpha,W_pos_norm*z(ii))*gamma(alpha+1)./((W_pos_norm*z(ii)/2).^(alpha));
    pos_sec_inside_div = besselj(alpha,W_pos_norm_col(1:sample_num)*z(ii))*gamma(alpha+1)./((W_pos_norm_col(1:sample_num)*z(ii)/2).^(alpha));
    neg_first_inside_div = besselj(alpha,W_neg_norm*z(ii))*gamma(alpha+1)./((W_neg_norm*z(ii)/2).^(alpha));
    neg_sec_inside_div = besselj(alpha,W_neg_norm_col(1:sample_num)*z(ii))*gamma(alpha+1)./((W_neg_norm_col(1:sample_num)*z(ii)/2).^(alpha));
    pos_first_inside_div_index=find(~isnan(pos_first_inside_div));
    pos_first_inside_div = pos_first_inside_div(pos_first_inside_div_index);
    pos_sec_inside_div_index=find(~isnan(pos_sec_inside_div));
    pos_sec_inside_div = pos_sec_inside_div(pos_sec_inside_div_index);    
    neg_first_inside_div_index=find(~isnan(neg_first_inside_div));
    neg_first_inside_div = neg_first_inside_div(neg_first_inside_div_index);
    neg_sec_inside_div_index=find(~isnan(neg_sec_inside_div));
    neg_sec_inside_div = neg_sec_inside_div(neg_sec_inside_div_index);
    
    mean_pos_first(ii) = mean(pos_first_inside_div);
    mean_pos_sec(ii) = mean(pos_sec_inside_div);
    mean_pos_sec(ii) = mean_pos_sec(ii)^2;
    mean_neg_first(ii) = mean(neg_first_inside_div);
    mean_neg_sec(ii) = mean(neg_sec_inside_div);
    mean_neg_sec(ii) = mean_neg_sec(ii)^2;
end
    
    
    
%% calculate the mean
mean_prodin = zeros(1,length(z));
mean_prodout = zeros(1,length(z));
% 
for ii = 1:length(z)
    W_pos = interp1(pos_cdf, w_total,random('unif',0,1,2*sample_num,1), 'linear', 0);
    W_neg = interp1(neg_cdf, w,random('unif',0,1,2*sample_num,1), 'linear', 0);
    W_pos = reshape(W_pos,sample_num,2);
    W_neg = reshape(W_neg,sample_num,2);
    cos_W_pos = cos(W_pos*z(ii));
    cos_W_neg = cos(W_neg*z(ii));;
    cos_W_posneg = cos_W_pos(:,1).*cos_W_neg(:,1);
    mean_pos_fir(ii) = mean(cos_W_pos(:,1));
    mean_neg_fir(ii) = mean(cos_W_neg(:,1));
    mean_prodin(ii) = mean(cos_W_posneg);
    mean_prodout(ii) = mean_pos_fir(ii)*mean_neg_fir(ii);
%     
end

F = mean_prodin - mean_prodout;
error = -(s-1)*kernelpower_pos_coeff^2*(mean_pos_first-mean_pos_sec)/s-(s-1)*kernelpower_neg_coeff^2*(mean_neg_first-mean_neg_sec)/s;
error = smooth(error);
errorIndex = find(z<0.035);
error(errorIndex) = 0;
error = smooth(error);
    
figure
plot(z,error)


