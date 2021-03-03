%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas

%%
clc;
clear all;
close all
%% load data only training;
rng(1234555555,'twister')

X = csvread('I:\Boxers_Study\Ctx_Analysis_04182020\ML\CTX+Volume\training.csv');
Y = csvread('I:\Boxers_Study\Ctx_Analysis_04182020\ML\CTX+Volume\Y_training.csv');
randomize = 0;
if randomize ==1
    % Randomize Y
    Y1=Y(randperm(length(Y)));
end
%
[~,Nf] = size(X);
%% Do Random sampling and find the features
index1 = find(Y==1);
index2 = find(Y==2);
N1 = length(index1);
N2 = length(index2);
X1 = X(index1,:);
X2 = X(index2,:);
Y1 = Y(index1,:);
Y2 = Y(index2,:);

N_sample1 = 1000;
N_sample2 = 1000;

mean_X1 = mean(X1);
cov_X1 = cov(X1);
sample_X1 = mvnrnd(mean_X1,cov_X1,N_sample1);
cov_X1_sample = cov(sample_X1);
mean_X1_sample = mean(sample_X1);

mean_X2 = mean(X2);
cov_X2 = cov(X2);
sample_X2 = mvnrnd(mean_X2,cov_X2,N_sample2);
cov_X2_sample = cov(sample_X2);
mean_X2_sample = mean(sample_X2);

norm((cov_X2-cov_X2_sample),'fro')
norm((cov_X1-cov_X1_sample),'fro')
X_sample = [sample_X1;sample_X2];
Y_sample = [ones(N_sample1,1);zeros(N_sample2,1)];

X_sample = standardization(X_sample);
%% feature selection;
subjectWeights = ones(N_sample1+N_sample2,1);
opts=struct('weights',subjectWeights);
glmnet_fitInfo = glmnet(X_sample,Y_sample,'binomial',opts);
glmnet_lambda = glmnet_fitInfo.lambda;
glmnet_beta = glmnet_fitInfo.beta;
%% Divide the original dataset into training and testing and CV test;  (0.6,0.2,0.2)
id_train_2 = randperm(length(Y2));
num_elements_training = ceil(0.8*length(id_train_2));
X_train_2 = X2(id_train_2(1:num_elements_training),:);
Y_train_2 = Y2(id_train_2(1:num_elements_training),:);
X_test_2 = X2(id_train_2(1+num_elements_training:end),:);
Y_test_2 = Y2(id_train_2(1+num_elements_training:end),:);

id_train_1 = randperm(length(Y1));
num_elements_training = ceil(0.8*length(id_train_1));
X_train_1 = X1(id_train_1(1:num_elements_training),:);
Y_train_1 = Y1(id_train_1(1:num_elements_training),:);
X_test_1 = X1(id_train_1(1+num_elements_training:end),:);
Y_test_1 = Y1(id_train_1(1+num_elements_training:end),:);

X_train = [X_train_2;X_train_1];
X_test = [X_test_2;X_test_1];
Y_train = [Y_train_2;Y_train_1];
Y_test = [Y_test_2;Y_test_1];

Y_train(Y_train==2) = 0;
Y_test(Y_test==2) = 0;
[N_train,~] = size(X_train);
[N_test,~] = size(X_test);
mean_train = mean(X_train);
std_train = std(X_train);
X_train = standardization(X_train);
X_test = X_test - ones(N_test,1)*mean_train;
X_test = X_test ./ (ones(N_test,1)*std_train + eps);


index11 = Y_train == 1;
index00 = Y_train == 0;
X_train_1 = X_train(index11,:);
X_train_2 = X_train(index00,:);
Y_train_1 = Y_train(index11,:);
Y_train_2 = Y_train(index00,:);
%% divide training set into training and cross-validation set, 
K_fold = 10; Num_iter = 10;
multiple_sigma =2;
N_group = 2;
% for K = 2:1:15
K = 7;
K
rbfn_parameter = struct('N_group',N_group,'K',K,'multiple_sigma',multiple_sigma);
machine = cell(Num_iter,1);
AUC_all = zeros(Num_iter,1);
feature_index_all = cell(Num_iter,1);
for iter = 1:Num_iter
    id_train_CV_2 = randperm(length(Y_train_2));
    num_elements_CV = ceil(0.2*length(id_train_CV_2));
    X_train_CV_2 = X_train_2(id_train_CV_2(1:num_elements_CV),:);
    Y_train_CV_2 = Y_train_2(id_train_CV_2(1:num_elements_CV),:);
    X_train_train_2 = X_train_2(id_train_CV_2(1+num_elements_CV:end),:);
    Y_train_train_2 = Y_train_2(id_train_CV_2(1+num_elements_CV:end),:);

    id_train_CV_1 = randperm(length(Y_train_1));
    num_elements_CV = ceil(0.2*length(id_train_CV_1));
    X_train_CV_1 = X_train_1(id_train_CV_1(1:num_elements_CV),:);
    Y_train_CV_1 = Y_train_1(id_train_CV_1(1:num_elements_CV),:);
    X_train_train_1 = X_train_1(id_train_CV_1(1+num_elements_CV:end),:);
    Y_train_train_1 = Y_train_1(id_train_CV_1(1+num_elements_CV:end),:);
    X_train_train = [X_train_train_2;X_train_train_1];
    Y_train_train = [Y_train_train_2;Y_train_train_1];
    X_train_CV = [X_train_CV_2;X_train_CV_1];
    Y_train_CV = [Y_train_CV_2;Y_train_CV_1];
    
    %% Cross Validation with training set itself select number of features;
    AUC_feature = zeros(length(glmnet_lambda),1);
    feature_index_t = cell(length(glmnet_lambda),1);
    for i = 1:length(glmnet_lambda)
        feature_index_t{i,1} = find(glmnet_beta(:,i)~=0);
        if (~isempty(feature_index_t{i,1})) && (~isequal(feature_index_t{i,1},feature_index_t{i-1,1}));
            N_feature_s = length(feature_index_t{i,1});
             AUC_feature(i,1) = CV_10fold_getfeature(K_fold,X_train_train,Y_train_train,feature_index_t{i,1},rbfn_parameter);
            if (AUC_feature(i,1)<=0.5) && (AUC_feature(i-1,1)<=0.5) && (~isempty(feature_index_t{i-1,1}))
                break;
            end
%             if (N_feature_s >= round(Nf/2))
%                 break;
%             end
        end
    end
    [~,ind_feature_index_f] = max(AUC_feature);
    feature_index_f = feature_index_t{ind_feature_index_f};
    X_train_train = X_train_train(:,feature_index_f);
    X_train_CV = X_train_CV(:,feature_index_f);
    [Y_vali1,rbf_structure] = RBFN_funcVer(X_train_train,X_train_CV,Y_train_train,N_group,K,multiple_sigma);
    machine{iter,1} = rbf_structure; 
    [~,~,~,AUC_all(iter,1)] = perfcurve(Y_train_CV,Y_vali1,1);
    % [~,~,RAW] = xlsread('F:\Boxer_Study\AllFeatureAlldataMachineLearning\data\0804\features_with_TOF_no_CBF_no_GM_diff.xlsx');
    feature_index_all{iter,1} = feature_index_f;
end
[AUC_max,ind_iter] = max(AUC_all);
feature_index_final = feature_index_all{ind_iter,1};

rbf_structure_f = machine{ind_iter,1};
save('Z:\Virendra\CTx_04222020\For_Sharing\ML_Output.mat','feature_index_final','rbf_structure_f');
clear all; close all;

