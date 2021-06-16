function X=HOSVD(A,i,n) %n is the dimension required
[U,S,V]=svd(mode_n_matricization(A,i),'econ');
X=U(:,1:n);
