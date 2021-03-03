%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%
function [Y_vali1,rbfnstr] = RBFN_funcVer(X_train,X_test,Y_train,N_group,K,multipleSigma)
% apply IPW during training;
% multipleSigma decides the scale of the sigma
    [N_train,~] = size(X_train);    
    [N_test,~] = size(X_test);
    index11=find(Y_train == 1);
    index22=find(Y_train == 0);
    N1_train = length(index11);
    N2_train = length(index22);
    D0 = zeros(N_train,N_group);
    P1 = (N_train/N1_train).^(1);
    P2 = (N_train/N2_train).^(1);
    D0(index11,1) = P1;
    D0(index22,2) = P2;
    rbfnstr = rbfn_rls(X_train,D0,K,multipleSigma);
    pd = rbfn_test(rbfnstr,X_test);
    Y_vali1 = pd(:,1);
end