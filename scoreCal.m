%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%

function s1 = scoreCal(Y_vali_f1,Y)
N0 = size(Y,1);
preAcc = sum(Y_vali_f1 == Y)/N0;
TP = sum(Y_vali_f1==1 & Y==1);
FP = sum(Y_vali_f1==1 & Y==0);
TN = sum(Y_vali_f1==0 & Y==0);
FN = sum(Y_vali_f1==0 & Y==1);
sensitivity = TP/(TP+FN);
specificity = TN/(TN+FP);
plotroc(Y',Y_vali_f1');
[~,~,~,AUC1] = perfcurve(Y,Y_vali_f1,1);
field1 = 'sensitivity'; field2 = 'specificy'; field3 = 'accuracy'; field4 = 'AUC';
s1 = struct(field1,sensitivity, field2,specificity,field3,preAcc,field4,AUC1);
end