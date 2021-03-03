%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%
function auc_f = CV_10fold_getfeature(K_fold,X_train,Y_train,feature_index_t,rbfn_parameter)
N_group = rbfn_parameter.N_group;
K = rbfn_parameter.K;
multiple_sigma = rbfn_parameter.multiple_sigma;
N_train = size(X_train,1);
Indices_train = crossvalind('Kfold', N_train, K_fold);
AUC = zeros(K_fold,1);
for k = 1:K_fold
    try
    train_test_index = (Indices_train == k);
    train_train_index = ~train_test_index;
    X_train_train = X_train(train_train_index,feature_index_t);
    X_train_test = X_train(train_test_index,feature_index_t);
    Y_train_train = Y_train(train_train_index,:);
    Y_train_test = Y_train(train_test_index,:);
    [Y_vali1,~] = RBFN_funcVer(X_train_train,X_train_test,Y_train_train,N_group,K,multiple_sigma);
    [~,~,~,AUC(k)] = perfcurve(Y_train_test,Y_vali1,1);
    plotroc(Y_train_test',Y_vali1');
    catch
        AUC(k) = 0.5;
    end
end
auc_f = mean(AUC);
end