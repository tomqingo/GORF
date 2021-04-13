clear all
clc

%% polynomial kernel difference
kernelparams.a = 3;
kernelparams.b = 1;
params.dim = 32;
compare_scale = 0.1;
s = params.dim;

z = 0:0.001:2;
ww = 10000;
num_points_w = 10000;
w = 0:ww/(2*num_points_w-1):ww;
dw = ww/(2*num_points_w-1);
kernelpower = zeros(1,length(w));
factor_b = factorial(kernelparams.b);

kernelfunc = @(normz) (1 - (normz/kernelparams.a).^2 ).^kernelparams.b;
a = kernelfunc(z);

for ii = 1:length(w)
    powertmp = zeros(1,kernelparams.b+1);
    if abs(2*w(ii))<sqrt(params.dim/2+1)*compare_scale
        for jj = 1:kernelparams.b+1
            jj = jj-1;
            factor_denominator = factorial(kernelparams.b-jj);
            coeff = (factor_b/factor_denominator)*((1-4/(kernelparams.a)^2)^(kernelparams.b-jj))*((2/(kernelparams.a)^2)^jj)*(2^(params.dim/2+jj))/gamma(params.dim/2+jj+1);
            powertmp(jj+1) = coeff;
        end
    else
        for jj = 1:kernelparams.b+1
            jj = jj-1;
            factor_denominator = factorial(kernelparams.b-jj);
            Jab = besselj(params.dim/2+jj,2*w(ii));
            coeff = (factor_b/factor_denominator)*((1-4/(kernelparams.a)^2)^(kernelparams.b-jj))*((2/(kernelparams.a)^2)^jj)*((2/w(ii))^(params.dim/2+jj));
            powertmp(jj+1) = coeff*Jab;
        end
    end
        
    kernelpower(ii) = (1/(2*pi))^(params.dim/2)*sum(powertmp);
end

kernelpower_pos = zeros(1,length(w));
kernelpower_neg = zeros(1,length(w));
kernelpower_pos(kernelpower>0) = kernelpower(kernelpower>0);
kernelpower_neg(kernelpower<0) = -kernelpower(kernelpower<0);
kernelpower_pos_coeff = abs(trapz(w,abs(kernelpower_pos)));
kernelpower_neg_coeff = abs(trapz(w,abs(kernelpower_neg)));
kernelpower_pos = kernelpower_pos/kernelpower_pos_coeff;
kernelpower_neg = kernelpower_neg/kernelpower_neg_coeff;

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
% for positive and negative parts, different measures are adopted
pos_cdf = cumsum(kernelpower_pos./sum(kernelpower_pos));
neg_cdf = cumsum(kernelpower_neg./sum(kernelpower_neg));
[pos_cdf, pos_uniq_ind] = unique(pos_cdf);
[neg_cdf, neg_uniq_ind] = unique(neg_cdf);
w_total = w;
w_total = w_total(pos_uniq_ind);
w = w(neg_uniq_ind);

kernelpower_pos_coeff = kernelpower_pos_coeff/(kernelpower_pos_coeff+kernelpower_neg_coeff);
kernelpower_neg_coeff = kernelpower_neg_coeff/(kernelpower_pos_coeff+kernelpower_neg_coeff);
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
    cos_W_neg = cos(W_neg*z(ii));
    cos_W_posneg = cos_W_pos(:,1).*cos_W_neg(:,1);
    mean_pos_fir(ii) = mean(cos_W_pos(:,1));
    mean_neg_fir(ii) = mean(cos_W_neg(:,1));
    mean_prodin(ii) = mean(cos_W_posneg);
    mean_prodout(ii) = mean_pos_fir(ii)*mean_neg_fir(ii);
%     
end

%G = (kernelpower_pos_coeff^2*mean_pos) - (kernelpower_neg_coeff^2*mean_neg);
%H = (kernelpower_pos_coeff^2*kernel_pos.^2) - (kernelpower_neg_coeff^2*kernel_neg.^2);
%G = (kernelpower_pos_coeff^2*mean_pos) - (kernelpower_pos_coeff^2*kernel_pos.^2);
%H = (kernelpower_neg_coeff^2*mean_neg) - (kernelpower_neg_coeff^2*kernel_neg.^2);
F = mean_prodin - mean_prodout;
%error = (s-1)*(G+H)/s+2*kernelpower_pos_coeff*kernelpower_neg_coeff*F;
error = -(s-1)*kernelpower_pos_coeff^2*(mean_pos_first-mean_pos_sec)/s-(s-1)*kernelpower_neg_coeff^2*(mean_neg_first-mean_neg_sec)/s;
% error = error+2*kernelpower_pos_coeff*kernelpower_neg_coeff*F;
error = smooth(error);
errorIndex = find(z<0.035);
error(errorIndex) = 0;
%error = G-H;
figure
plot(z,error)

