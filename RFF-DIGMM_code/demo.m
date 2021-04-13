close all
clear all
%addpath('D:\liblinear-2.20\matlab\');
%rand('state',0);    randn('state',0);
times_num = 10;
res_ours_test = zeros(1,times_num);
res_RBF_test = zeros(1,times_num);
trtime = zeros(1,times_num);
trtime_RFF = zeros(1,times_num);
flagCV = 0;% cross validation for C

load('letter.mat')

%[Y_train,X_train] = libsvmread('.\ijcnn1.train');
%[Y_test,X_test] = libsvmread('.\ijcnn1.t');
X_train = X_train./repmat(sqrt(sum(X_train.^2,2)),1,size(X_train,2));
X_test = X_test./repmat(sqrt(sum(X_test.^2,2)),1,size(X_test,2));

X= 0, y= 0;

kernelfunc = @(normz) (1 - (normz/3).^2 ).^(1);
for sss = 1:times_num
    
    N=size(X_train,1);
    d = size(X_train,2);
    ntest=size(X_test,1);
    
    M =64*d; %feature dimension
    Nsample =10;
    l = size(X_train,1);
    T=5; %D=50; % T: maximum # of GPs; Mupper: maximum # of active set points for one GP
    
    %timebegin1 = cputime;
    D = M;
    G = randn(d,M);b = rand(1,M)*2*pi;
    Omega = G';% transformation matrix
    
    sto_times = 5; rho = 1;
    qwmstomu =cell(M,1);qwmstosigma = cell(M,1);
    timebegin = cputime;
    for iit = 1:sto_times
        p = randperm(l);
        % p = 1:l;
        TrainX = X_train(p(1:Nsample),:);
        TrainY = Y_train(p(1:Nsample));
        
        ktrain = zeros(Nsample, Nsample);
        for i = 1:Nsample
            for j = 1:Nsample
                ktrain(i,j) = (1 - (norm(TrainX(i,:)-TrainX(j,:))/3).^2 ).^1; % polynomial kernel on the unit sphere
                %ktrain(i,j) = exp(norm(TrainX(i,:)-TrainX(j,:)).^2/2)-exp(norm(TrainX(i,:)-TrainX(j,:)).^2/(2*10^2));
            end
        end
        
        eigvmin = min(eig(ktrain));
        if eigvmin < 0
            [U,S] = eig(ktrain);
            kernel_train1 = U*(S-20*eigvmin*eye(size(S,1),size(S,1)))*U';
            kernel_train2 = U*(-20*eigvmin*eye(size(S,1),size(S,1)))*U';
        else
            kernel_train1 = ktrain;
            kernel_train2 = 0;
        end
        %================================================Variational
        %inference
        
        for in_num  = 1:2
            
            if in_num == 1
                ktrain = kernel_train1;
            else
                ktrain = kernel_train2;
            end
            
            
            %---initialize the parameters and hyperparameters
            alpha0=1;
            SigmaX=cov(Omega); CholFa=chol(SigmaX); Rx=CholFa\(CholFa'\eye(d));
            m0=zeros(d,1);  gamma0 = 1; R0 = Rx;
            vareps = 0.01;
            nu0=d;   W0=Rx/d;    W0inv=d*SigmaX;
            temp=[1 ones(1,d) 0.05]; %[sigma_f sigma_1 ... sigma_d sigma_b]
            thetabold=repmat(temp, T, 1);
            ActCandSize=100;         %randomly pick 100 examples as the actset candidate
            ActCand=ones(T,ActCandSize);
            for ii=1:T
                temp=randperm(N);
                ActCand(ii,:)=temp(1:ActCandSize);
            end
            
            %--intialize the random feature
            % Z_train = createRandomFourierFeatures(M, G, b, X_train');
            % Z_test = createRandomFourierFeatures(M, G, b, X_test');
            
            
            %---k-means clustering, and initialize qzi
            [IDX, Centroid]=kmeans(Omega,T);
            IDXT=cell(T,1); ActSet=cell(T,1); qzi=zeros(T,M);
            for ii=1:T
                IDXT{ii}=find(IDX==ii);
                if length(IDXT{ii}) < (D-0.5)
                    temp=randperm(M);
                    IDXT{ii}=temp(1:D);
                end
                temp=randperm(length(IDXT{ii}));
                ActSet{ii}=IDXT{ii}(temp(1:D));
                
                qzi(ii,IDXT{ii})=1;
            end
            qzi=qzi./(eps+repmat(sum(qzi,1),T,1)); %posterior
            
            %---initialize other posterior distributions
            qbetak=zeros(T-1,2); qmuk=cell(T,1); qLambdak=cell(T,1);
            for ii=1:T
                if ii<T
                    temp=qzi(ii+1:T, :);
                    sumqzgk=sum(temp(:)); %\sum q(z_m > t)
                    sumqzek=sum(qzi(ii, :)); % \sum q(z_m=t)
                    qbetak(ii,:)=[sumqzek+1, sumqzgk+alpha0]; %posterior
                end
                
                %-------compute Lambda_k--------------
                sumqzek=sum(qzi(ii,:)); % \sum q(z_m=t)
                mutemp=(qzi(ii,:)*Omega)'; % \sum q(z_m=t)w_m    维度R^d
                omegabar =  mutemp/(eps+sumqzek); % \bar(w_m)
                
                sumqzkwm = zeros(d,d); % sum q(z_m=k) w_m w_m'   Wk的第二项
                for m = 1:M
                    temp = (Omega(m,:)' - omegabar)*(Omega(m,:)' - omegabar)';
                    sumqzkwm = sumqzkwm + qzi(ii,m)*temp;
                end
                %gqzkw = gamma0* sumqzek/(eps+gamma0+sumqzek)*(omegabar-m0)*(omegabar-m0)'; % Wk的第三项
                qLambdak{ii}.nu=nu0+sumqzek;               %posterior
                
                %compute Wishart
                %         CholFa=chol(W0inv+sumqzkwm);
                %         qLambdak{ii}.W=CholFa\(CholFa'\eye(d)); %posterior
                qLambdak{ii}.W=inv(W0inv+sumqzkwm);
                %---------------------------------------------------
                
                %------compute mu_k
                EWk = qLambdak{ii}.nu * qLambdak{ii}.W;
                qmuk{ii}.R=R0 + EWk*sumqzek;                 %posterior
                %CholFa=chol(qmuk{ii}.R); Rkinv=CholFa\(CholFa'\eye(d));
                Rkinv = inv(qmuk{ii}.R);
                qmuk{ii}.Sigma=Rkinv;
                
                qmuk{ii}.mu=qmuk{ii}.Sigma*(R0*m0+EWk*mutemp); %posterior %   \sum q(z_m=t)E(w_m)    R^d
            end
            
            %----------compute w_m
            qwm=cell(M,1);
            for m = 1:M
                s1 = 0; s3 = 0;
                for ii = 1:T
                    Emuk = qmuk{ii}.mu;
                    ELambdak =  qLambdak{ii}.nu*qLambdak{ii}.W;
                    s1 = s1+ qzi(ii,m)*ELambdak;
                    s3 = s3 + qzi(ii,m)*ELambdak*Emuk;
                end
                s2 = zeros(d,d);
                for i =1:Nsample
                    for j = 1:Nsample
                        s2 = s2 + 1/2/vareps/vareps*(1-ktrain(i,j))*(TrainX(i,:) - TrainX(j,:))*(TrainX(i,:) - TrainX(j,:))';
                    end
                end
                qwm{m}.R = s1 + s2; %posterior
                %CholFa=chol(qwm{m}.R); Rminv=CholFa\(CholFa'\eye(d));
                Rminv = inv(qwm{m}.R);
                qwm{m}.Sigma=Rminv;
                qwm{m}.mu = Rminv*s3; %posterior
            end
            
            
            %---if active sets during two EM processes change, do another EM process
            MaxOuterIteration=1;%maximum # of updating active sets
            for outerstep=1:MaxOuterIteration
                %disp(['OuterStep = ' num2str(outerstep) '...']);
                
                MaxEMite=1;
                qzidiffmat = [];qwmdiffmat = [];
                for EMite=1:MaxEMite
                    %             disp(['..EM iteration= ' num2str(EMite) '...']);
                    %----------iteratively update the variational distribution: E-step
                    MaxIteration_VI=50; %maximum number of variational inference during one E-step
                    for VInum=1:MaxIteration_VI
                        disp(['...updating variational posterior...VInum= ' num2str(VInum) '...']);
                        tempa=psi(qbetak(:,1)); tempb=psi(qbetak(:,2)); tempab=psi(qbetak(:,1)+qbetak(:,2));
                        Eln_beta=tempa-tempab; Eln_beta=[Eln_beta; 0]; %T rows, 1 column
                        Eln_oneminusnut=tempb-tempab;               %T-1 rows, 1 column
                        lnront=zeros(T,M);
                        for ii=1:T
                            if ii==1
                                first4parts=Eln_beta(ii) + (sum(psi((qLambdak{ii}.nu+1-[1:d])/2))+d*log(2)+log(det(qLambdak{ii}.W)) + d )/2;
                            else
                                first4parts=Eln_beta(ii) + sum(Eln_oneminusnut(1:ii-1)) + (sum(psi((qLambdak{ii}.nu+1-[1:d])/2))+d*log(2)+log(det(qLambdak{ii}.W)) + d )/2;
                            end
                            lnront(ii,:)=first4parts;
                        end
                        lnront=exp(lnront);
                        qziold=qzi;
                        qzi=lnront./(eps+repmat(sum(lnront,1),T,1)); %posterior
                        qzidiff=qziold-qzi;
                        qzidiffmat =[qzidiffmat,norm(qzidiff,'fro')];
                        if VInum>1 & norm(qzidiff,'fro')<1e-5
                            qzi=qziold;
                            break; %finish updating variational posterior
                        end
                        %=update other posterior distributions
                        for ii=1:T
                            if ii<T
                                temp=qzi(ii+1:T, :);
                                sumqzgk=sum(temp(:)); sumqzek=sum(qzi(ii, :));
                                qbetak(ii,:)=[sumqzek+1 sumqzgk+alpha0]; %posterior
                            end
                            %-------compute Lambda_k--------------
                            sumqzek=sum(qzi(ii,:)); % \sum q(z_m=t)
                            mutemp=(qzi(ii,:)*Omega)'; % \sum q(z_m=t)w_m    维度R^d
                            omegabar =  mutemp/(eps+sumqzek); % \bar(w_m)
                            
                            sumqzkwm = zeros(d,d); % sum q(z_m=k) w_m w_m'   Wk的第二项
                            for m = 1:M
                                temp = (Omega(m,:)' - omegabar)*(Omega(m,:)' - omegabar)';
                                sumqzkwm = sumqzkwm + qzi(ii,m)*temp;
                            end
                            %gqzkw = gamma0* sumqzek/(eps+gamma0+sumqzek)*(omegabar-m0)*(omegabar-m0)'; % Wk的第三项
                            qLambdak{ii}.nu=nu0+sumqzek;               %posterior
                            
                            %compute Wishart
                            %                 CholFa=chol(W0inv+sumqzkwm);
                            %                 qLambdak{ii}.W=CholFa\(CholFa'\eye(d)); %posterior
                            qLambdak{ii}.W = inv(W0inv+sumqzkwm);
                            %---------------------------------------------------
                            
                            %------compute mu_k
                            EWk = qLambdak{ii}.nu * qLambdak{ii}.W;
                            qmuk{ii}.R=R0 + EWk*sumqzek;                 %posterior
                            %CholFa=chol(qmuk{ii}.R); Rkinv=CholFa\(CholFa'\eye(d));
                            Rkinv = inv(qmuk{ii}.R);
                            qmuk{ii}.Sigma=Rkinv;
                            
                            qmuk{ii}.mu=qmuk{ii}.Sigma*(R0*m0+EWk*mutemp); %posterior %注意这里应该是   \sum q(z_m=t)E(w_m)    维度R^d
                            %---------------------------------------------------
                        end
                        
                        %----------compute w_m
                        qwm=cell(M,1);
                        for m = 1:M
                            s1 = 0; s3 = 0;
                            for ii = 1:T
                                Emuk = qmuk{ii}.mu;
                                ELambdak =  qLambdak{ii}.nu*qLambdak{ii}.W;
                                s1 = s1+ qzi(ii,m)*ELambdak;
                                s3 = s3 + qzi(ii,m)*ELambdak*Emuk;
                            end
                            s2 = zeros(d,d);
                            for i =1:Nsample
                                for j = 1:Nsample
                                    s2 = s2 + 1/2/vareps/vareps*(1-ktrain(i,j))*(TrainX(i,:) - TrainX(j,:))*(TrainX(i,:) - TrainX(j,:))';
                                end
                            end
                            qwm{m}.R = s1 + s2; %posterior
                            %CholFa=chol(qwm{m}.R); Rminv=CholFa\(CholFa'\eye(d));
                            %qwm{m}.Sigma=Rminv;
                            qwm{m}.Sigma=inv(qwm{m}.R);
                            qwm{m}.mu = Rminv*s3; %posterior
                            
                            
                            
                            if iit > 1
                                qwmstomu{m} = rho*qwm{m}.mu + (1-rho)*temp1;
                                qwmstosigma{m} = rho*qwm{m}.Sigma + (1-rho)*temp2;
                            else
                                qwmstomu{m} = rho*qwm{m}.mu;
                                qwmstosigma{m} = rho*qwm{m}.Sigma;
                            end
                            temp1 = qwm{m}.mu;   temp2 = qwm{m}.Sigma;
                        end
                        %qwmdiffmat =[qwmdiffmat,norm(qwm{m}.mu)];
                    end
                end
                
            end
            if in_num == 1
                qwmstomu1 = qwmstomu;
                qwmstosigma1 = qwmstosigma;
            else
                qwmstomu2 = qwmstomu;
                qwmstosigma2 = qwmstosigma;
            end
            
            if kernel_train2 == 0
                qwmstomu2 = 0;
                qwmstosigma2 = 0;
                break;
            end
            
        end
        rho = (iit + 1)^(-1);
    end
    trainingtime = cputime - timebegin;
    trtime(sss) = trainingtime;
    
    Gpred = zeros(d,M);
    
    for m = 1:M
        %Gpred(:,m) = mvnrnd(qwm{m}.mu,qwm{m}.Sigma);
        qwmstosigma1{m} = (qwmstosigma1{m} + qwmstosigma1{m}')/2;
        qwmstosigma1{m} = real(qwmstosigma1{m});
        if kernel_train2 == 0
            Gpred(:,m) = mvnrnd(qwmstomu1{m},qwmstosigma1{m}+1e-4*eye(size(qwmstosigma1{m},1)));
        else
            qwmstosigma2{m} = (qwmstosigma2{m} + qwmstosigma2{m}')/2;
            qwmstosigma2{m} = real(qwmstosigma2{m});
            Gpred(:,m) = mvnrnd(qwmstomu1{m},qwmstosigma1{m}+1e-4*eye(size(qwmstosigma1{m},1))) - mvnrnd(qwmstomu2{m},qwmstosigma2{m}+1e-4*eye(size(qwmstosigma2{m},1)));
        end
    end
    
    Z_trainours = createRandomFourierFeatures(M, Gpred, b, X_train');
    Z_testours = createRandomFourierFeatures(M, Gpred, b, X_test');
    
    kernel_gt = kernelfunc(sqrt(2-X_train*X_train'*2));
    kernel_est = Z_trainours'*Z_trainours;
    temp = ones(1,size(kernel_est,1));
    kernel_est = kernel_est - diag(diag(kernel_est)) + diag(temp);
    kernel_diff = norm(kernel_gt-kernel_est,'fro')/norm(kernel_gt,'fro');
    fprintf('Approx Error:%.2f\n', kernel_diff)
    
    Zx = sparse(Z_trainours');  Zt = sparse(Z_testours');
    
    
    if flagCV == 1
        L_fold = 5;
        [bestc, ~, ~] = automaticParameterSelectionliblinear(Y_train, Zx, L_fold);
        C = bestc;
    else
        C = 1;
    end
    libsvmparam1 = ['-s 2 -c ',num2str(C) ];
    
    
    
    svmodel = train(Y_train,Zx,libsvmparam1);
    [~,accuracy_ours_CV_test, ~] = predict( Y_test,  Zt, svmodel);
    
    
    %res_RBF_test(sss) =      acctest(1);
    res_ours_test(sss) = accuracy_ours_CV_test(1);
end
fprintf('Ours Approximation: Acc=%.2f±%.2f\n',mean(res_ours_test),std((res_ours_test)));
fprintf('training time=%.3f±%.3fs\n', mean(trtime),std(trtime));