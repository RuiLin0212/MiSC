clear all
load ./data/BM.mat;

I=find(groundtruth~=0);
label_matrix=label_matrix(:,I);
groundtruth=groundtruth(I);

Model = crowd_model(double(label_matrix'), 'true_labels',double(groundtruth));

% Majority voting:
mv = MajorityVote_crowd_model(Model);

% Dawid & Sknene:
ds = DawidSkene_crowd_model(Model);

% Set parameters:
lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
% 1. Categorical minimax entropy:
% result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:});
% 2. Ordinal minimax entropy:
result2 =  MinimaxEntropy_crowd_model(Model,'algorithm','ordinal',opts{:});

% Mean Field(DSMF)
Key_MFAB11 = variationalEM_two_coin_crowd_model(Model, 'ell', [2,1;1,2], 'maxIter',100, 'TOL', 1e-3);
prob_error_MFAB11 = mean(Key_MFAB11.ans_labels ~= groundtruth);
fprintf('Error rate of mean field two coin: %f\n', prob_error_MFAB11);


[Nw,Ni]=size(label_matrix);
Nc=double(max(label_matrix(:)));

T=zeros(Nw+1,Ni,Nc);
for s=1:Nc
    T(1:Nw,:,s)=(label_matrix==s);
end
for i=1:Ni
    T(Nw+1,i,result2.ans_labels(i))=1;
end

%--------------------------
%         TenALS
%--------------------------

picture = T;

[n1,n2,n3]=size(picture);
r = 10;
tol   = [];
nitr  = 100;
ninit = [];

error_rate = [];

itr = 1;
A = T;

% find the place that need to be completed
T_flatten = reshape(T,[(Nw+1) * Ni, Nc]);
Omega = ones(size(T_flatten));

for k = 1:((Nw+1) * Nc)
    if sum(T_flatten(k,:)) == 0
        Omega(k,:) = 0;
    end
end

Omega = reshape(Omega,[(Nw+1), Ni, Nc]);

E = Omega;

[V1 V2 V3 S dist] = TenALS(A, E, r, 20, nitr, tol);
sim=zeros(size(picture));
for i=1:r
    sim=sim+S(i)*reshape(kron(V3(:,i),kron(V2(:,i),V1(:,i))),size(picture));
end

simA = sim;

score=zeros(Nw,Ni);
for i=1:Nw+1
    [~,I]=max(reshape(simA(i,:),[size(simA,2),Nc])');
    score(i,:)=I;
end

Model = crowd_model(score', 'true_labels',double(groundtruth));

result2 =  MinimaxEntropy_crowd_model(Model,'algorithm','ordinal',opts{:});

error_rate1 = result2.error_rate;

T(end,:,:)=simA(end,:,:);