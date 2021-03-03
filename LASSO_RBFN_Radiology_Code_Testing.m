%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas

clc;clear all; close all;

%% Test here
pathfile = 'I:\Boxers_Study\Ctx_Analysis_04182020\ML\CTX+Volume';
load(fullfile(pathfile,'ML_Output.mat'));

X_test = csvread(fullfile(pathfile,'testing.csv'));
Y_test = csvread(fullfile(pathfile,'Y_testing.csv'));
Y_test(find(Y_test==2)) = 0;

X_test1 = X_test(:,feature_index_final);
Y_vali_f = rbfn_test(rbf_structure_f,X_test1);
clear X_test1;
Y_vali_f = Y_vali_f(:,1);
s1 = scoreCal(Y_vali_f,Y_test)
