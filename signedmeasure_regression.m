% Given kernel parameter a, b, dim, num_gaussians etc., generate
% the model file. Note that the model is independent of the data. 

%addpath(genpath('./minFunc_2012'));
clc
clear all;
close all;

%% Initialize parameters
% Kernel parameters. The kernel has the form:
% (1 - ||x - y||^2 / a^2 ) ^ b
load('L:/SRF/datasets/housing/housing.mat')
addpath(genpath('RFF-DIGMM_code'))
per = 0.2; % percentage for validation
% shuffle data
randIndex = randperm(size(housing,1));
housing = housing(randIndex,:);

num_sample = size(housing,1);
attr = size(housing,2);
num_train = num_sample - round(num_sample*0.2);
num_test = round(num_sample*0.2);

X_data = housing(:,1:attr-1);
Y_data = housing(:,attr);

% normalize
for ii = 1:size(X_data,2)
    X_data(:,ii) = (X_data(:,ii)-min(X_data(:,ii)))/(max(X_data(:,ii))-min(X_data(:,ii)));
end

X_train = X_data(1:num_train,:);
X_test = X_data(num_train+1:end,:);
Y_train = Y_data(1:num_train);
Y_test = Y_data(num_train+1:end);


%% normalize_data
params.dim = size(X_train,2);

% normalization
X_train_norm = sqrt(sum(X_train.^2,2));
X_test_norm = sqrt(sum(X_test.^2,2));
X_train = X_train./repmat(X_train_norm,1,size(X_train,2));
X_test = X_test./repmat(X_test_norm,1,size(X_test,2));

N_sample = 1000;
mapdim_index = [0,1,2,3];
mapdim_all = round(2.^(mapdim_index)*params.dim);

kernelparams.a = 3;
kernelparams.b = 1;
kernelparams.p = kernelparams.b;
kernelparams.alpha = (2/kernelparams.a^2)^(kernelparams.b);
kernelparams.q = kernelparams.a^2/2-1;

% Meta parameters for SRF
params.num_gaussians = 10; % Too many makes it hard to optimize, too few and it's not expressive enough.
params.poss = 2;
params.num_points_w = 10000; % Likewise.
params.eps = 1e-20;
kernelfunc = @(normz) (1 - (normz/kernelparams.a).^2 ).^kernelparams.b;

%% calculate the fourier transform of dot kernel 
compare_scale = 0.1;
ww = 100;
w = 0:ww/(2*params.num_points_w-1):ww;
kernelpower = zeros(1,length(w));
factor_b = factorial(kernelparams.b);

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
        
    kernelpower(ii) = (2*pi)^(-params.dim/2)*sum(powertmp);
end

kernelpower_coeff = trapz(w,abs(kernelpower));
kernelpower_nor = kernelpower/kernelpower_coeff;

%figure
%plot(w,kernelpower_nor)

%% divide mu+ and mu-
kernelpower_pos = zeros(1,length(w));
kernelpower_neg = zeros(1,length(w));
kernelpower_pos(kernelpower>0) = kernelpower(kernelpower>0);
kernelpower_neg(kernelpower<0) = -kernelpower(kernelpower<0);
kernelpower_pos_coeff = abs(trapz(w,abs(kernelpower_pos)));
kernelpower_neg_coeff = abs(trapz(w,abs(kernelpower_neg)));
kernelpower_pos = kernelpower_pos/kernelpower_pos_coeff;
kernelpower_neg = kernelpower_neg/kernelpower_neg_coeff;
kernelpower_coeff = kernelpower_pos_coeff+kernelpower_neg_coeff;
kernelpower_pos_coeff = kernelpower_pos_coeff/kernelpower_coeff;
kernelpower_neg_coeff = kernelpower_neg_coeff/kernelpower_coeff;

%% calculate the cumulative distribution
% for positive and negative parts, different measures are adopted
pos_cdf = cumsum(kernelpower_pos./sum(kernelpower_pos));
neg_cdf = cumsum(kernelpower_neg./sum(kernelpower_neg));
[pos_cdf, pos_uniq_ind] = unique(pos_cdf);
[neg_cdf, neg_uniq_ind] = unique(neg_cdf);
w_total = w;
w_total = w_total(pos_uniq_ind);
w = w(neg_uniq_ind);

%% calculate the probability density
n_lim = round(-log(params.eps)/log(params.poss)-1);
n = 0:1:n_lim;
probability_density = (1/params.poss).^(n+1);
probability_total = sum(probability_density);
probability_density = probability_density/probability_total;

probability_cdf = cumsum(probability_density);

%% calculate TS params
alpha_sub = kernelparams.alpha^(1/kernelparams.p);
bias_para = sqrt(alpha_sub*kernelparams.q);
coeff_para = sqrt(alpha_sub);
X_train_trans = [X_train*coeff_para,ones(num_train,1)*bias_para];
X_test_trans = [X_test*coeff_para,ones(num_test,1)*bias_para];

%% generate random features and evaluate
%% 5-cross-validation for kernel approximation
validation_times = 10;
MSE = zeros(length(mapdim_all), validation_times);
timer = zeros(length(mapdim_all), validation_times);
acc_train = zeros(length(mapdim_all), validation_times);
acc_test = zeros(length(mapdim_all), validation_times);
MSE_srf = zeros(length(mapdim_all), validation_times);
timer_srf = zeros(length(mapdim_all), validation_times);
acc_train_srf = zeros(length(mapdim_all), validation_times);
acc_test_srf = zeros(length(mapdim_all), validation_times);
MSE_orf = zeros(length(mapdim_all), validation_times);
timer_orf = zeros(length(mapdim_all), validation_times);
acc_train_orf = zeros(length(mapdim_all), validation_times);
acc_test_orf = zeros(length(mapdim_all), validation_times);
MSE_rmf = zeros(length(mapdim_all), validation_times);
timer_rmf = zeros(length(mapdim_all), validation_times);
acc_train_rmf = zeros(length(mapdim_all), validation_times);
acc_test_rmf = zeros(length(mapdim_all), validation_times);
MSE_tsf = zeros(length(mapdim_all), validation_times);
timer_tsf = zeros(length(mapdim_all), validation_times);
acc_train_tsf = zeros(length(mapdim_all), validation_times);
acc_test_tsf = zeros(length(mapdim_all), validation_times);

num_gaussians = 10;
typeC = 'liblinear';
flagCV = 0;

%X_train_new = [X_train;X_test];

for ii = 1:length(mapdim_all)
    mapdim = mapdim_all(ii);
    
    for jj = 1:validation_times
            
         % GRFF MSE
         [grff_train, grff_test, timer(ii,jj)] = gen_GRFF(X_train, X_test, pos_cdf, neg_cdf, kernelpower_pos_coeff, kernelpower_neg_coeff, w_total, w, params.dim, mapdim);
         
          % GORF MSE
         [gorf_train, gorf_test, timer_orf(ii,jj)] = gen_GORF_mix(X_train, X_test, pos_cdf, neg_cdf, kernelpower_pos_coeff, kernelpower_neg_coeff, w_total, w, params.dim, mapdim);
         
          % SRF MSE
         [rff_srf_train, rff_srf_test, timer_srf(ii,jj)] = gen_RFF(X_train, X_test, kernelparams.a, kernelparams.b, num_gaussians, mapdim);
         
          %RM MSE
          [rmf_train, rmf_test, timer_rmf(ii,jj)] = gen_RMF(X_train, X_test, kernelparams.alpha, kernelparams.q, kernelparams.p, params.poss, mapdim, probability_cdf, n);
         
          %TS MSE
          [tsf_train, tsf_test, timer_tsf(ii,jj)] = gen_TSF(X_train_trans, X_test_trans, mapdim, params.dim);
          
           
           grff_train = grff_train.*repmat(X_train_norm',size(grff_train,1),1);
           grff_test = grff_test.*repmat(X_test_norm',size(grff_test,1),1);
           gorf_train = gorf_train.*repmat(X_train_norm',size(grff_train,1),1);
           gorf_test = gorf_test.*repmat(X_test_norm',size(grff_test,1),1);
           rff_srf_train = rff_srf_train.*repmat(X_train_norm',size(rff_srf_train,1),1);
           rff_srf_test = rff_srf_test.*repmat(X_test_norm',size(rff_srf_test,1),1);
           rmf_train = rmf_train.*repmat(X_train_norm',size(rmf_train,1),1);
           rmf_test = rmf_test.*repmat(X_test_norm',size(rmf_test,1),1);
           tsf_train = tsf_train.*repmat(X_train_norm',size(tsf_train,1),1);
           tsf_test = tsf_test.*repmat(X_test_norm',size(tsf_test,1),1);
            
%         
         %classification
         [acc_train(ii,jj), acc_test(ii,jj), outGRFF] = RFregressionRKKS(grff_train,grff_test,Y_train,Y_test, typeC,flagCV);
         [acc_train_orf(ii,jj), acc_test_orf(ii,jj), outGORF] = RFregressionRKKS(gorf_train,gorf_test,Y_train,Y_test, typeC,flagCV);         
         [acc_train_srf(ii,jj), acc_test_srf(ii,jj), outSRF] = RFregressionRKKS(rff_srf_train,rff_srf_test,Y_train,Y_test, typeC,flagCV);             
         [acc_train_rmf(ii,jj), acc_test_rmf(ii,jj), outRMF] = RFregressionRKKS(rmf_train,rmf_test,Y_train,Y_test, typeC,flagCV);
         [acc_train_tsf(ii,jj), acc_test_tsf(ii,jj), outTSF] = RFregressionRKKS(tsf_train,tsf_test,Y_train,Y_test, typeC,flagCV);
    end
end


% % accuracy
ACC_train_avg = zeros(length(mapdim_all),1);
ACC_train_std = zeros(length(mapdim_all),1);
ACC_test_avg = zeros(length(mapdim_all),1);
ACC_test_std = zeros(length(mapdim_all),1);
ACC_train_avg_srf = zeros(length(mapdim_all),1);
ACC_train_std_srf = zeros(length(mapdim_all),1);
ACC_test_avg_srf = zeros(length(mapdim_all),1);
ACC_test_std_srf = zeros(length(mapdim_all),1);
ACC_train_avg_orf = zeros(length(mapdim_all),1);
ACC_train_std_orf = zeros(length(mapdim_all),1);
ACC_test_avg_orf = zeros(length(mapdim_all),1);
ACC_test_std_orf = zeros(length(mapdim_all),1);
ACC_train_avg_rmf = zeros(length(mapdim_all),1);
ACC_train_std_rmf = zeros(length(mapdim_all),1);
ACC_test_avg_rmf = zeros(length(mapdim_all),1);
ACC_test_std_rmf = zeros(length(mapdim_all),1);
ACC_train_avg_tsf = zeros(length(mapdim_all),1);
ACC_train_std_tsf = zeros(length(mapdim_all),1);
ACC_test_avg_tsf = zeros(length(mapdim_all),1);
ACC_test_std_tsf = zeros(length(mapdim_all),1);

for ii = 1:length(mapdim_all)
    ACC_train_avg(ii) = mean(acc_train(ii,:));
    ACC_train_std(ii) = std(acc_train(ii,:));
    ACC_test_avg(ii) = mean(acc_test(ii,:));
    ACC_test_std(ii) = std(acc_test(ii,:));
    ACC_train_avg_srf(ii) = mean(acc_train_srf(ii,:));
    ACC_train_std_srf(ii) = std(acc_train_srf(ii,:));
    ACC_test_avg_srf(ii) = mean(acc_test_srf(ii,:));
    ACC_test_std_srf(ii) = std(acc_test_srf(ii,:));
    ACC_train_avg_orf(ii) = mean(acc_train_orf(ii,:));
    ACC_train_std_orf(ii) = std(acc_train_orf(ii,:));
    ACC_test_avg_orf(ii) = mean(acc_test_orf(ii,:));
    ACC_test_std_orf(ii) = std(acc_test_orf(ii,:));
    ACC_train_avg_rmf(ii) = mean(acc_train_rmf(ii,:));
    ACC_train_std_rmf(ii) = std(acc_train_rmf(ii,:));
    ACC_test_avg_rmf(ii) = mean(acc_test_rmf(ii,:));
    ACC_test_std_rmf(ii) = std(acc_test_rmf(ii,:));
    ACC_train_avg_tsf(ii) = mean(acc_train_tsf(ii,:));
    ACC_train_std_tsf(ii) = std(acc_train_tsf(ii,:));
    ACC_test_avg_tsf(ii) = mean(acc_test_tsf(ii,:));
    ACC_test_std_tsf(ii) = std(acc_test_tsf(ii,:));
end

figure;
hold on
h1=errorbar(mapdim_index, ACC_train_avg_rmf, ACC_train_std_rmf,'-*c');
h2=errorbar(mapdim_index, ACC_train_avg_tsf, ACC_train_std_tsf,'-or');
h3=errorbar(mapdim_index, ACC_train_avg_srf, ACC_train_std_srf,'-*r');
h5=errorbar(mapdim_index, ACC_train_avg, ACC_train_std,'-ob');
h6=errorbar(mapdim_index, ACC_train_avg_orf, ACC_train_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('RMSE')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h5,h6],'GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

figure;
hold on
h1=errorbar(mapdim_index, ACC_test_avg_rmf, ACC_test_std_rmf,'-*c');
h2=errorbar(mapdim_index, ACC_test_avg_tsf, ACC_test_std_tsf,'-or');
h3=errorbar(mapdim_index, ACC_test_avg_srf, ACC_test_std_srf,'-*r');
h5=errorbar(mapdim_index, ACC_test_avg, ACC_test_std,'-ob');
h6=errorbar(mapdim_index, ACC_test_avg_orf, ACC_test_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('RMSE')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h5,h6],'GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

figure;
hold on
h1=plot(Y_test,'-ok');
h2=plot(outSRF,'-*r');
h4=plot(outGORF,'-ob');
xlabel('n');
ylabel('housing prices')
legend([h1,h2,h3,h4],'True','SRF','DIGMM','GORF(ours)','orientation','horizontal','location','north');
xlim([0,30])
grid on
hold off