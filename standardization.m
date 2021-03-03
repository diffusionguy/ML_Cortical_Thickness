%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%

function X = standardization(X)
    tdim = size(X,1);
    X = X - ones(tdim,1)*mean(X);
    X = X ./(ones(tdim,1)*std(X)+eps);
end