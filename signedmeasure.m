% Given kernel parameter a, b, dim, num_gaussians etc., generate
% the model file. Note that the model is independent of the data. 

%addpath(genpath('./minFunc_2012'));
clc
clear all;
close all;

%% Initialize parameters
% Kernel parameters. The kernel has the form:
% (1 - ||x - y||^2 / a^2 ) ^ b;
addpath(genpath('RFF-DIGMM_code'))
load('P:\SRF\datasets\letter\letter.mat');
num_train = size(X_train,1);
num_test = size(X_test,1);


%% normalize_data
params.dim = size(X_train,2);

% normalization
X_train = X_train./repmat(sqrt(sum(X_train.^2,2)),1,size(X_train,2));
X_test = X_test./repmat(sqrt(sum(X_test.^2,2)),1,size(X_test,2));

N_sample = 1000;
mapdim_index = [-2,-1,0,1,2,3];
mapdim_all = round(2.^(mapdim_index)*params.dim);

kernelparams.a = 3;
kernelparams.b = 1;
kernelparams.p = kernelparams.b;
kernelparams.alpha = (2/kernelparams.a^2)^(kernelparams.b);
kernelparams.q = kernelparams.a^2/2-1;

params.poss = 2;
params.num_points_w = 10000;
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


figure
plot(w,kernelpower_pos)

figure
plot(w,kernelpower_neg)
    

%% calculate the cumulative distribution
% for positive and negative parts, different measures are adopted
%w_neg = -ww:ww/(2*params.num_points_x-1):-0.01;
%w_total = [w_neg,w];
%kernelpower_pos = [fliplr(kernelpower_pos),kernelpower_pos];
pos_cdf = cumsum(kernelpower_pos./sum(kernelpower_pos));
neg_cdf = cumsum(kernelpower_neg./sum(kernelpower_neg));
[pos_cdf, pos_uniq_ind] = unique(pos_cdf);
[neg_cdf, neg_uniq_ind] = unique(neg_cdf);
w_total = w;
w_total = w_total(pos_uniq_ind);
w = w(neg_uniq_ind);
figure
plot(w_total, pos_cdf)

figure
plot(w,neg_cdf)

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
MSE_dimm = zeros(length(mapdim_all)-3, validation_times);
timer_dimm = zeros(length(mapdim_all)-3, validation_times);
acc_train_dimm = zeros(length(mapdim_all)-3, validation_times);
acc_test_dimm = zeros(length(mapdim_all)-3, validation_times);

num_gaussians = 10;
typeC = 'liblinear';
flagCV = 0;

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
          
          %DIGMM MSE
           if ii>3
             [digmm_train, digmm_test, timer_dimm(ii-3,jj)] = gen_DIGMM(X_train, X_test, kernelparams.a, kernelparams.b, mapdim);
             [acc_train_dimm(ii-3,jj), acc_test_dimm(ii-3,jj)] = RFclassificationRKKS(digmm_train,digmm_test,Y_train,Y_test, typeC, flagCV);
          end
                  
         %classification
         [acc_train(ii,jj), acc_test(ii,jj)] = RFclassificationRKKS(grff_train,grff_test,Y_train,Y_test, typeC,flagCV);
         [acc_train_orf(ii,jj), acc_test_orf(ii,jj)] = RFclassificationRKKS(gorf_train,gorf_test,Y_train,Y_test, typeC,flagCV);         
         [acc_train_srf(ii,jj), acc_test_srf(ii,jj)] = RFclassificationRKKS(rff_srf_train,rff_srf_test,Y_train,Y_test, typeC,flagCV);             
         [acc_train_rmf(ii,jj), acc_test_rmf(ii,jj)] = RFclassificationRKKS(rmf_train,rmf_test,Y_train,Y_test, typeC,flagCV);
         [acc_train_tsf(ii,jj), acc_test_tsf(ii,jj)] = RFclassificationRKKS(tsf_train,tsf_test,Y_train,Y_test, typeC,flagCV);
         
         if(size(X_train,1)>N_sample)
            rand_r = randperm(size(X_train,1));
            X_sub = X_train(rand_r(1:N_sample),:);
            grff_sub = grff_train(:,rand_r(1:N_sample));
            gorf_sub = gorf_train(:,rand_r(1:N_sample));
            rff_srf_sub = rff_srf_train(:,rand_r(1:N_sample));
            rmf_sub = rmf_train(:,rand_r(1:N_sample));
            tsf_sub = tsf_train(:,rand_r(1:N_sample));
            if ii>3
                digmm_sub = digmm_train(:,rand_r(1:N_sample));
            end
         end
            
         kernel_gt = kernelfunc(sqrt(2-X_sub*X_sub'*2));
         
         kernel_predict = conj(grff_sub')*grff_sub;
         temp = ones(1,size(kernel_predict,1));
         kernel_predict = kernel_predict - diag(diag(kernel_predict)) + diag(temp);
         kernel_diff = norm(kernel_gt-kernel_predict,'fro')/norm(kernel_gt,'fro');
         MSE(ii,jj) = mean(kernel_diff(:));
        
         kernel_predict_gorf = conj(gorf_sub')*gorf_sub;
         temp = ones(1,size(kernel_predict_gorf,1));
         kernel_predict_gorf = kernel_predict_gorf - diag(diag(kernel_predict_gorf)) + diag(temp);
         kernel_diff_gorf = norm(kernel_gt-kernel_predict_gorf,'fro')/norm(kernel_gt,'fro');
         MSE_orf(ii,jj) = mean(kernel_diff_gorf(:));
        
         kernel_predict_srf = conj(rff_srf_sub')*rff_srf_sub;
         temp = ones(1,size(kernel_predict_srf,1));
         kernel_predict_srf = kernel_predict_srf - diag(diag(kernel_predict_srf)) + diag(temp);
         kernel_diff_srf = norm(kernel_gt-kernel_predict_srf,'fro')/norm(kernel_gt,'fro');
         MSE_srf(ii,jj) = mean(kernel_diff_srf(:));
         
         kernel_predict_rmf = conj(rmf_sub')*rmf_sub;
         temp = ones(1,size(kernel_predict_rmf,1));
         kernel_predict_rmf = kernel_predict_rmf - diag(diag(kernel_predict_rmf)) + diag(temp);
         kernel_diff_rmf = norm(kernel_gt-kernel_predict_rmf,'fro')/norm(kernel_gt,'fro');
         MSE_rmf(ii,jj) = mean(kernel_diff_rmf(:));   
         
         kernel_predict_tsf = conj(tsf_sub')*tsf_sub;
         temp = ones(1,size(kernel_predict_tsf,1));
         kernel_predict_tsf = kernel_predict_tsf - diag(diag(kernel_predict_tsf)) + diag(temp);
         kernel_diff_tsf = norm(kernel_gt-kernel_predict_tsf,'fro')/norm(kernel_gt,'fro');
         MSE_tsf(ii,jj) = mean(kernel_diff_tsf(:));
         
         if ii>3
            kernel_predict_digmm = conj(digmm_sub')*digmm_sub;
            temp = ones(1,size(kernel_predict_digmm,1));
            kernel_predict_digmm = kernel_predict_digmm - diag(diag(kernel_predict_digmm)) + diag(temp);
            kernel_diff_digmm = norm(kernel_gt-kernel_predict_digmm,'fro')/norm(kernel_gt,'fro');
            MSE_dimm(ii-3,jj) = mean(kernel_diff_digmm(:));
         end
         
         
     end
end

%mapdim_index = [0,1,2,3,4];
%mapdim_all = 2.^(mapdim_index)*params.dim;
% mse
MSE_avg = zeros(length(mapdim_all),1);
MSE_std = zeros(length(mapdim_all),1);
MSE_avg_srf = zeros(length(mapdim_all),1);
MSE_std_srf = zeros(length(mapdim_all),1);
MSE_avg_orf = zeros(length(mapdim_all),1);
MSE_std_orf = zeros(length(mapdim_all),1);
MSE_avg_rmf = zeros(length(mapdim_all),1);
MSE_std_rmf = zeros(length(mapdim_all),1);
MSE_avg_tsf = zeros(length(mapdim_all),1);
MSE_std_tsf = zeros(length(mapdim_all),1);
MSE_avg_digmm = zeros(length(mapdim_all)-3,1);
MSE_std_digmm = zeros(length(mapdim_all)-3,1);

for ii = 1:length(mapdim_all)
    MSE_avg(ii) = mean(MSE(ii,:));
    MSE_std(ii) = std(MSE(ii,:));
    MSE_avg_srf(ii) = mean(MSE_srf(ii,:));
    MSE_std_srf(ii) = std(MSE_srf(ii,:));
    MSE_avg_orf(ii) = mean(MSE_orf(ii,:));
    MSE_std_orf(ii) = std(MSE_orf(ii,:));
    MSE_avg_rmf(ii) = mean(MSE_rmf(ii,:));
    MSE_std_rmf(ii) = std(MSE_rmf(ii,:));
    MSE_avg_tsf(ii) = mean(MSE_tsf(ii,:));
    MSE_std_tsf(ii) = std(MSE_tsf(ii,:));
    if ii>3
    MSE_avg_digmm(ii-3) = mean(MSE_dimm(ii-3,:));
    MSE_std_digmm(ii-3) = std(MSE_dimm(ii-3,:));
    end
end

figure;
hold on
h1=errorbar(mapdim_index, MSE_avg_rmf, MSE_std_rmf,'-*c');
h2=errorbar(mapdim_index, MSE_avg_tsf, MSE_std_tsf,'-or');
h3=errorbar(mapdim_index, MSE_avg_srf, MSE_std_srf,'-*r');
h4=errorbar(mapdim_index(4:end), MSE_avg_digmm, MSE_std_digmm,'-*g');
h5=errorbar(mapdim_index, MSE_avg, MSE_std,'-ob');
h6=errorbar(mapdim_index, MSE_avg_orf, MSE_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('approximation error')
%legend([h5,h6],'GRFF','GORF(Ours)')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h4,h5,h6],'DIGMM','GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

%time
TIME_avg = zeros(length(mapdim_all),1);
TIME_std = zeros(length(mapdim_all),1);
TIME_avg_srf = zeros(length(mapdim_all),1);
TIME_std_srf = zeros(length(mapdim_all),1);
TIME_avg_orf = zeros(length(mapdim_all),1);
TIME_std_orf = zeros(length(mapdim_all),1);
TIME_avg_rmf = zeros(length(mapdim_all),1);
TIME_std_rmf = zeros(length(mapdim_all),1);
TIME_avg_tsf = zeros(length(mapdim_all),1);
TIME_std_tsf = zeros(length(mapdim_all),1);
TIME_avg_digmm = zeros(length(mapdim_all)-3,1);
TIME_std_digmm = zeros(length(mapdim_all)-3,1);


for ii = 1:length(mapdim_all)
    TIME_avg(ii) = mean(timer(ii,:));
    TIME_std(ii) = std(timer(ii,:));
    TIME_avg_srf(ii) = mean(timer_srf(ii,:));
    TIME_std_srf(ii) = std(timer_srf(ii,:));
    TIME_avg_orf(ii) = mean(timer_orf(ii,:));
    TIME_std_orf(ii) = std(timer_orf(ii,:));
    TIME_avg_rmf(ii) = mean(timer_rmf(ii,:));
    TIME_std_rmf(ii) = std(timer_rmf(ii,:));
    TIME_avg_tsf(ii) = mean(timer_tsf(ii,:));
    TIME_std_tsf(ii) = std(timer_tsf(ii,:));
    if ii>3
    TIME_avg_digmm(ii-3) = mean(timer_dimm(ii-3,:));
    TIME_std_digmm(ii-3) = std(timer_dimm(ii-3,:));
    end
end

figure;
hold on
h1=errorbar(mapdim_index, TIME_avg_rmf, TIME_std_rmf,'-*c');
h2=errorbar(mapdim_index, TIME_avg_tsf, TIME_std_tsf,'-or');
h3=errorbar(mapdim_index, TIME_avg_srf, TIME_std_srf,'-*r');
h4=errorbar(mapdim_index(4:end), TIME_avg_digmm, TIME_std_digmm,'-*g');
h5=errorbar(mapdim_index, TIME_avg, TIME_std,'-ob');
h6=errorbar(mapdim_index, TIME_avg_orf, TIME_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('time(sec)')
% legend([h5,h6],'GRFF','GORF(Ours)')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h4,h5,h6],'DIGMM','GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

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
ACC_train_avg_digmm = zeros(length(mapdim_all)-3,1);
ACC_train_std_digmm = zeros(length(mapdim_all)-3,1);
ACC_test_avg_digmm = zeros(length(mapdim_all)-3,1);
ACC_test_std_digmm = zeros(length(mapdim_all)-3,1);

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
    if ii>3
    ACC_train_avg_digmm(ii-3) = mean(acc_train_dimm(ii-3,:));
    ACC_train_std_digmm(ii-3) = std(acc_train_dimm(ii-3,:));
    ACC_test_avg_digmm(ii-3) = mean(acc_test_dimm(ii-3,:));
    ACC_test_std_digmm(ii-3) = std(acc_test_dimm(ii-3,:));
    end


end

figure;
hold on
h1=errorbar(mapdim_index, ACC_train_avg_rmf, ACC_train_std_rmf,'-*c');
h2=errorbar(mapdim_index, ACC_train_avg_tsf, ACC_train_std_tsf,'-or');
h3=errorbar(mapdim_index, ACC_train_avg_srf, ACC_train_std_srf,'-*r');
h4=errorbar(mapdim_index(4:end), ACC_train_avg_digmm, ACC_train_std_digmm,'-*g');
h5=errorbar(mapdim_index, ACC_train_avg, ACC_train_std,'-ob');
h6=errorbar(mapdim_index, ACC_train_avg_orf, ACC_train_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('classification accuracy')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h4,h5,h6],'DIGMM','GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

figure;
hold on
h1=errorbar(mapdim_index, ACC_test_avg_rmf, ACC_test_std_rmf,'-*c');
h2=errorbar(mapdim_index, ACC_test_avg_tsf, ACC_test_std_tsf,'-or');
h3=errorbar(mapdim_index, ACC_test_avg_srf, ACC_test_std_srf,'-*r');
h4=errorbar(mapdim_index(4:end), ACC_test_avg_digmm, ACC_test_std_digmm,'-*g');
h5=errorbar(mapdim_index, ACC_test_avg, ACC_test_std,'-ob');
h6=errorbar(mapdim_index, ACC_test_avg_orf, ACC_test_std_orf,'--og');
xlabel('log_2(s/d)');
ylabel('classification accuracy')
legend([h1,h2,h3],'RM','TS','SRF','orientation','horizontal','location','north');
legend boxoff
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,[h4,h5,h6],'DIGMM','GRFF','GORF(Ours)','orientation','horizontal','location','north');
legend boxoff
grid on
hold off

