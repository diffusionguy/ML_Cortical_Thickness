%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%

function rbfnstr=rbfn_rls(x,dd,K,multipleSigma)
%Radial-basis function (RBF) networks using recursive least-squares
%x: input data N-by-p; dd: indecies N-by-q (-1 and 1, so decision boundary 
%is at 0); K: # of hidden nodes; w: weights of the output layer; cen: 
%cetroids of RBF; sigma: constant for Gaussian function
%1) Suggest running 10 times and picking the smallest MSE due to the randomness of
%K-means clustering!
%2) Suitable for q output nodes (for more than 2 groups) achieved by LS of output nodes.
%e.g. three groups: [1 -1 -1], [-1 1 -1], [-1 -1 1]
% Mingwu Jin, UTA, Jan 27, 2014
% Revised Sept 24, 2014: 1) binary coding nominal dd (1 0); 2) seperate clustering for
% different classes-->actual cluster # 2K; 3) varaible width (sigma) if

%convert class labels d (nominal array) to binary
d=dd;

%1. K-means clustering and RBF definition
[idx,cen,sumD]=kmeans(x,K,'emptyaction','singleton','Replicates',100);
CK1 = max(cen,[],2);
if sum(isinf(CK1))>0
    K = K - sum(isinf(CK1));
    [idx,cen,sumD] = kmeans(x,K,'emptyaction','singleton','Replicates',100);
end

%either the same or estimated from the clusters?

bf=pdist2(x,cen);%N*K
[N,p]=size(x);

%xiaowei's method for sigma
sumd = zeros(K,1);
sigma = zeros(1,K);
for i= 1:K
    num = nnz(idx == i);
    index = find(idx == i);
    sum1 = 0;
    for j = 1:num
        sum1 = sum1 + norm(x(index(j),:)-cen(i,:));
    end
    sumd(i) = sum1;
    sigma(i) = sumd(i)/num;
end
% mingwu's method for sigma;
% sigma=sumD';
% for k=1:K
%     sigma(k)=sigma(k)/nnz(idx==k);
% end
sigma = sigma .* multipleSigma;
sigma=sigma.^2;
sigma(sigma==0)=1e-10;
phi=exp(-bf.^2/2./repmat(sigma,N,1));
q=size(d,2);
w=zeros(K+1,q);
mse=zeros(q,1);
phi=[ones(size(phi,1),1) phi];
for n=1:q %train for each output node
    ccv=d(:,n); %cross-correlation vector: K*1
    w(:,n)=pinv(phi)*ccv;
    mse(n)=mean((d(:,n)-phi*w(:,n))'*(d(:,n)-phi*w(:,n)));
end
rbfnstr=struct('w',w,'centroid',cen,'sigma',sigma,'disToCen',bf,'phi',phi,'cat',size(d,2),'MSE',mse);