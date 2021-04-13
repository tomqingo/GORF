function [accTrain, accTest] = RFclassificationRKKS(Z_train,Z_test,Y_train,Y_test, type,flagCV)

switch type
    case 'lr'
        meany = mean(Y_train);
        if flagCV == 1
            L_fold = 5;
            [bestc, ~, ~] = automaticParameterSelectionlambda(Y_train, Z_train, L_fold);
            lambda = bestc;
        else
            %lambda = 1e-4;
            lambda = .1;
            %lambda = 1e-12;
        end
        w = (Z_train * conj(Z_train') + lambda * eye(size(Z_train,1))) \ (Z_train * (Y_train-meany));
        
        [err,~, ~] = computeErrorRKKS(Z_train, w, meany, Y_train);
        accTrain = (1-err)*100;
        
        [err,~, ~] = computeErrorRKKS(Z_test, w, meany, Y_test);
        accTest = (1-err)*100;
    case 'liblinear'
        Z_train(all(Z_train==0,2),:) = [];
        Z_test(all(Z_test==0,2),:) = [];
        Zx = sparse(conj(Z_train'));  Zt = sparse(conj(Z_test'));
%         Zx = conj(Z_train');
%         Zt = conj(Z_test');
        if flagCV == 1
            L_fold = 5;
            [bestc, ~, ~] = automaticParameterSelectionliblinear(Y_train, Zx, L_fold);
            C = bestc;
        else
            C = 1000;
        end
        
        libsvmparam1 = ['-s 2 -c ',num2str(C) ];
        svmodel = train(Y_train,Zx,libsvmparam1);
        
        [~,accuracyTr, ~] = predict( Y_train,  Zx, svmodel);
        [~,accuracyTe, ~] = predict( Y_test,  Zt, svmodel);
        accTrain = accuracyTr(1);
        accTest = accuracyTe(1);
end
        
