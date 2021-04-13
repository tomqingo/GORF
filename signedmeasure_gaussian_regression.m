clear all
clc

%% load the dataset
load('L:/SRF/datasets/housing/housing.mat')

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


dim = size(X_train,2);

X_train_norm = sqrt(sum(X_train.^2,2));
X_test_norm = sqrt(sum(X_test.^2,2));
X_train = X_train./repmat(X_train_norm,1,size(X_train,2));
X_test = X_test./repmat(X_test_norm,1,size(X_test,2));

N_sample = 1000;
mapdim_index = [-2,-1,0,1,2,3];
mapdim_all = round(2.^(mapdim_index)*dim);

sigma_fir = 1;
sigma_sec = 10;
num_gaussians = 3;
typeC = 'liblinear';
flagCV = 0;

kernelfunc = @(normz) (exp(-(normz.^2)/(2*sigma_fir^2))-exp(-(normz.^2)/(2*sigma_sec^2)));

validation_times = 10;
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
        
        grff_train = grff_train.*repmat(X_train_norm',size(grff_train,1),1);
        grff_test = grff_test.*repmat(X_test_norm',size(grff_test,1),1);
        gorf_train = gorf_train.*repmat(X_train_norm',size(grff_train,1),1);
        gorf_test = gorf_test.*repmat(X_test_norm',size(grff_test,1),1);
        srf_train = srf_train.*repmat(X_train_norm',size(srf_train,1),1);
        srf_test = srf_test.*repmat(X_test_norm',size(srf_test,1),1);
        % GRFF classification
        [acc_train_RFF(ii,jj), acc_test_RFF(ii,jj),~] = RFregressionRKKS(grff_train, grff_test, Y_train, Y_test, typeC, flagCV);
        [acc_train_ORF(ii,jj), acc_test_ORF(ii,jj),~] = RFregressionRKKS(gorf_train, gorf_test, Y_train, Y_test, typeC, flagCV);
        [acc_train_SRF(ii,jj), acc_test_SRF(ii,jj),~] = RFregressionRKKS(srf_train, srf_test, Y_train, Y_test, typeC, flagCV);        
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