%% Developer: Xiaowei Zhuang, Imaging Research, Cleveland Clinic Las Vegas
%%
function pd=rbfn_test(rbfnstr,x_test)
%predicted indices of test samples using the traing results from rbfn_rls.

w=rbfnstr.w;
cen=rbfnstr.centroid;
sigma=rbfnstr.sigma;
catnum=rbfnstr.cat;

bf=pdist2(x_test,cen);%N*K
if length(sigma)==1
    phi=exp(-bf/2/sigma);
    phi=phi./repmat(sum(phi,2),1,size(phi,2));
else
    phi=exp(-bf/2./repmat(sigma,size(bf,1),1));
end

phi=[ones(size(phi,1),1) phi];
pd=zeros(size(x_test,1),catnum);
if catnum==1
    pd=phi*w(:)>.5;
    pd=nominal(double(pd));
else
    temp=phi*w;
    for n=1:size(x_test,1)
        m=find(temp(n,:)==max(temp(n,:)));
        pd(n,m(1))=1;%assign 1 for the largest score in 'catnum' categories
        
    end
%     pd=nominal(pd);
end