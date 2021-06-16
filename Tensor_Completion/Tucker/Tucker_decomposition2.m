function [B,e]=Tucker_decomposition2(A,R,varargin) %R=[R_1:R_d]
a=size(A);
d=length(a);% d-way tensor
if ~isempty(varargin)
    r=varargin{1};
else
    r=R;
end
for i=1:d
    B{i}=HOSVD(A,i,r(i)); 
end
impr=100;
for i=1:1000 %iteration steps
    if impr>1e-8 %improvement scale
        for n=1:d
            Y=A;
            for j=[1:n-1,n+1:d]
                Y=k_mode_product(Y,(B{j})',j);
            end
            B{n}=HOSVD(Y,n,R(n));
        end
        simA=k_mode_product(Y,B{d}',d);
        for j=1:d
            simA=k_mode_product(simA,B{j},j);
        end
        e(i)=norm(simA(:)-A(:))/norm(A(:));
        if i==1
            impr=e(1);
        else
            impr=e(i-1)-e(i);
        end
    else
        break;
    end
end
i
B{d+1}=A; %S=B{d+1}
for k=1:d
    B{d+1}=k_mode_product(B{d+1},(B{k})',k);
end
end

