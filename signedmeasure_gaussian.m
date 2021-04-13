clear all
clc

%% load the dataset

load('L:\SRF\datasets\letter\letter.mat')
addpath(genpath('L:\SRF\RFF-DIGMM_code'))

num_train = size(X_train,1);
num_test = size(X_test,1);


dim = size(X_train,2);

X_train = X_train./repmat(sqrt(sum(X_train.^2,2)),1,size(X_train,2));
X_test = X_test./repmat(sqrt(sum(X_test.^2,2)),1,size(X_test,2));


N_sample = 1000;
mapdim_index = [-2,-1,0,1,2,3];
mapdim_all = round(2.^(mapdim_index)*dim);

sigma_fir = 1;
sigma_sec = 10;
num_gaussians = 5;
typeC = 'liblinear';
flagCV = 0;

kernelfunc = @(normz) (exp(-(normz.^2)/(2*sigma_fir^2))-exp(-(normz.^2)/(2*sigma_sec^2)));

validation_times = 10;
MSE_SRF = zeros(length(mapdim_all), validation_times);
MSE_RFF = zeros(length(mapdim_all), validation_times);
MSE_ORF = zeros(length(mapdim_all), validation_times);
TIME_SRF = zeros(length(mapdim_all), validation_times);
TIME_RFF = zeros(length(mapdim_all), validation_times);
TIME_ORF = zeros(length(mapdim_all), validation_times);
acc_train_SRF = zeros(length(mapdim_all), validation_times);
acc_test_SRF = zeros(length(mapdim_all), validation_times);
acc_train_RFF = zeros(length(mapdim_all), validation_times);
acc_test_RFF = zeros(length(mapdim_all), validation_times);
acc_train_ORF = zeros(length(mapdim_all), validation_times);
acc_test_ORF = zeros(length(mapdim_all), validation_times);


for ii = 1:length(mapdim_all)
    mapdim = mapdim_all(ii);
    
    for jj = 1:validation_times
        
        % GRFF MSE
        [grff_train, grff_test, TIME_RFF(ii,jj)] = gen_GRFF_gaussian(X_train, X_test, sigma_fir, sigma_sec, mapdim, dim);
        % GORF MSE
        [gorf_train, gorf_test, TIME_ORF(ii,jj)] = gen_GORF_gaussian_mix(X_train, X_test, sigma_fir, sigma_sec, mapdim, dim);
        % SRF MSE
        [srf_train, srf_test, TIME_SRF(ii,jj)] = gen_RFF_gaussian(X_train, X_test, sigma_fir, sigma_sec, num_gaussians, mapdim);
        
        % GRFF classification
        [acc_train_RFF(ii,jj), acc_test_RFF(ii,jj)] = RFclassificationRKKS(grff_train, grff_test, Y_train, Y_test, typeC, flagCV);
        [acc_train_ORF(ii,jj), acc_test_ORF(ii,jj)] = RFclassificationRKKS(gorf_train, gorf_test, Y_train, Y_test, typeC, flagCV);
        [acc_train_SRF(ii,jj), acc_test_SRF(ii,jj)] = RFclassificationRKKS(srf_train, srf_test, Y_train, Y_test, typeC, flagCV);
        
        if(size(X_train,1)>N_sample)
            rand_r = randperm(size(X_train,1));
            X_sub = X_train(rand_r(1:N_sample),:);
            grff_sub = grff_train(:,rand_r(1:N_sample));
            gorf_sub = gorf_train(:,rand_r(1:N_sample));
            srf_sub = srf_train(:,rand_r(1:N_sample));
        end
        
        kernel_gt = kernelfunc(sqrt(2-X_sub*X_sub'*2));
        kernel_predict = conj(grff_sub')*grff_sub;
        temp = zeros(1,size(kernel_predict,1));
        kernel_predict = kernel_predict - diag(diag(kernel_predict)) + diag(temp);
        kernel_diff = norm(kernel_gt-kernel_predict,'fro')/norm(kernel_gt,'fro');
        MSE_RFF(ii,jj) = mean(kernel_diff(:));

        kernel_predict = conj(gorf_sub')*gorf_sub;
        temp = zeros(1,size(kernel_predict,1));
        kernel_predict = kernel_predict - diag(diag(kernel_predict)) + diag(temp);
        kernel_diff = norm(kernel_gt-kernel_predict,'fro')/norm(kernel_gt,'fro');
        MSE_ORF(ii,jj) = mean(kernel_diff(:));
        
        kernel_predict = conj(srf_sub')*srf_sub;
        temp = ones(1,size(kernel_predict,1));
        kernel_predict = kernel_predict - diag(diag(kernel_predict)) + diag(temp);
        kernel_predict = kernel_predict - 1;
        kernel_diff = norm(kernel_gt-kernel_predict,'fro')/norm(kernel_gt,'fro');
        MSE_SRF(ii,jj) = mean(kernel_diff(:));
        
                
    end
end

%mse
MSE_avg_srf = zeros(length(mapdim_all),1);
MSE_std_srf = zeros(length(mapdim_all),1);
MSE_avg_rff = zeros(length(mapdim_all),1);
MSE_std_rff = zeros(length(mapdim_all),1);
MSE_avg_orf = zeros(length(mapdim_all),1);
MSE_std_orf = zeros(length(mapdim_all),1);

for ii = 1:length(mapdim_all)
    MSE_avg_srf(ii) = mean(MSE_SRF(ii,:));
    MSE_std_srf(ii) = std(MSE_SRF(ii,:));
    MSE_avg_rff(ii) = mean(MSE_RFF(ii,:));
    MSE_std_rff(ii) = std(MSE_RFF(ii,:));
    MSE_avg_orf(ii) = mean(MSE_ORF(ii,:));
    MSE_std_orf(ii) = std(MSE_ORF(ii,:));
end

figure;
hold on
errorbar(mapdim_index, MSE_avg_srf, MSE_std_srf, '-og');
errorbar(mapdim_index, MSE_avg_rff, MSE_std_rff,'-ob');
errorbar(mapdim_index, MSE_avg_orf, MSE_std_orf,'--or');
xlabel('log_2(s/d)');
ylabel('approximation error')
legend('SRF','GRFF','GORF(Ours)','Location','north','Orientation','horizontal')
legend boxoff
hold off
grid on


%time
TIME_avg_srf = zeros(length(mapdim_all),1);
TIME_std_srf = zeros(length(mapdim_all),1);
TIME_avg_rff = zeros(length(mapdim_all),1);
TIME_std_rff = zeros(length(mapdim_all),1);
TIME_avg_orf = zeros(length(mapdim_all),1);
TIME_std_orf = zeros(length(mapdim_all),1);

for ii = 1:length(mapdim_all)
    TIME_avg_srf(ii) = mean(TIME_SRF(ii,:));
    TIME_std_srf(ii) = std(TIME_SRF(ii,:));
    TIME_avg_rff(ii) = mean(TIME_RFF(ii,:));
    TIME_std_rff(ii) = std(TIME_RFF(ii,:));
    TIME_avg_orf(ii) = mean(TIME_ORF(ii,:));
    TIME_std_orf(ii) = std(TIME_ORF(ii,:));
end

figure;
hold on
errorbar(mapdim_index, TIME_avg_srf, TIME_std_srf, '-og');
errorbar(mapdim_index, TIME_avg_rff, TIME_std_rff,'-ob');
errorbar(mapdim_index, TIME_avg_orf, TIME_std_orf,'--or');
xlabel('log_2(s/d)');
ylabel('time(sec)')
legend('SRF','GRFF','GORF(Ours)','Location','north','Orientation','horizontal')
legend boxoff
hold off
grid on
        
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

for ii = 1:length(mapdim_all)
    ACC_train_avg(ii) = mean(acc_train_RFF(ii,:));
    ACC_train_std(ii) = std(acc_train_RFF(ii,:));
    ACC_test_avg(ii) = mean(acc_test_RFF(ii,:));
    ACC_test_std(ii) = std(acc_test_RFF(ii,:));
    ACC_train_avg_srf(ii) = mean(acc_train_SRF(ii,:));
    ACC_train_std_srf(ii) = std(acc_train_SRF(ii,:));
    ACC_test_avg_srf(ii) = mean(acc_test_SRF(ii,:));
    ACC_test_std_srf(ii) = std(acc_test_SRF(ii,:));
    ACC_train_avg_orf(ii) = mean(acc_train_ORF(ii,:));
    ACC_train_std_orf(ii) = std(acc_train_ORF(ii,:));
    ACC_test_avg_orf(ii) = mean(acc_test_ORF(ii,:));
    ACC_test_std_orf(ii) = std(acc_test_ORF(ii,:));
end

figure;
hold on
errorbar(mapdim_index, ACC_train_avg_srf, ACC_train_std_srf,'-og');
errorbar(mapdim_index, ACC_train_avg, ACC_train_std,'-ob');
errorbar(mapdim_index, ACC_train_avg_orf, ACC_train_std_orf,'--or');
xlabel('log_2(s/d)');
ylabel('classification accuracy')
legend('SRF','GRFF','GORF(Ours)','Location','north','Orientation','horizontal')
legend boxoff
hold off
grid on

figure;
hold on
errorbar(mapdim_index, ACC_test_avg_srf, ACC_test_std_srf,'-og');
errorbar(mapdim_index, ACC_test_avg, ACC_test_std,'-ob');
errorbar(mapdim_index, ACC_test_avg_orf, ACC_test_std_orf,'--or');
xlabel('log_2(s/d)');
ylabel('classification accuracy')
legend('SRF','GRFF','GORF(Ours)','Location','north','Orientation','horizontal')
legend boxoff
hold off
grid on


    
