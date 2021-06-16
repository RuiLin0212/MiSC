function X=mode_n_matricization(A,n)
a=size(A);
b=a(n);
c=prod(a)/b;
X=reshape(permute(A,[n,1:n-1,n+1:length(a)]),[b,c]);
