function X=k_mode_product(A,U,k)
a=size(A);
d=length(a);% d-way tensor
%K=a(k);% the dimension of kth direction
X=mode_n_matricization(A,k);
X=U*X;
a(k)=size(U,1);%the size of the new tensor
b=[a(k),a(1:k-1),a(k+1:d)];
X=reshape(X,b);
X=permute(X,[2:k,1,k+1:d]);
