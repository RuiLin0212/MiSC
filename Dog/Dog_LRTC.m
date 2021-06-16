clear all
load ./data/dog.mat;

Model = crowd_model(double(label_matrix'), 'true_labels',double(groundtruth));

% Majority voting:
mv = MajorityVote_crowd_model(Model);

% Dawid & Sknene:
ds = DawidSkene_crowd_model(Model);

% Mean Field(DSMF)
Key_MFAB11 = variationalEM_two_coin_crowd_model(Model, 'ell', [2,1,1,1;1,2,1,1;1,1,2,1;1,1,1,2], 'maxIter',100, 'TOL', 1e-3);
prob_error_MFAB11 = mean(Key_MFAB11.ans_labels ~= groundtruth);
fprintf('Error rate of mean field two coin: %f\n', prob_error_MFAB11);

% Set parameters:
lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
% 1. Categorical minimax entropy:
result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:});
% 2. Ordinal minimax entropy:
% result2 =  MinimaxEntropy_crowd_model(Model,'algorithm','ordinal',opts{:});

[Nw,Ni]=size(label_matrix);
Nc=max(label_matrix(:));

T=zeros(Nw+1,Ni,Nc);
for s=1:Nc
    T(1:Nw,:,s)=(label_matrix==s);
end
for i=1:Ni
    T(Nw+1,i,ds.ans_labels(i))=1;
end

% ----------------------------------------
%               Simple LRTC
% ----------------------------------------
% find the place that need to be completed
T_flatten = reshape(T,[(Nw+1) * Ni, Nc]);
Omega = ones(size(T_flatten));

for k = 1:((Nw+1) * Nc)
    if sum(T_flatten(k,:)) == 0
        Omega(k,:) = 0;
    end
end

Omega = logical(reshape(Omega,[(Nw+1), Ni, Nc]));

% setting parameteres for LRTC
alpha = [1, 1, 0];
alpha = alpha / sum(alpha);
maxIter = 500;
epsilon = 1e-6;
beta = 0.1*ones(1, ndims(T));

error_rate = [];

itr = 1;

[X_S, errList_S] = SiLRTC(...
    T,...                      % a tensor whose elements in Omega are used for estimating missing value
    Omega,...           % the index set indicating the obeserved elements
    alpha, ...             % the coefficient of the objective function,  i.e., \|X\|_* := \alpha_i \|X_{i(i)}\|_*
    beta,...                % the relaxation parameter. The larger, the closer to the original problem. See the function for definitions.
    maxIter,...         % the maximum iterations
    epsilon...            % the tolerance of the relative difference of outputs of two neighbor iterations
    );

simA = X_S;

score=zeros(Nw,Ni);
for i=1:Nw+1
    [~,I]=max(reshape(simA(i,:),[size(simA,2),4])');
    score(i,:)=I;
end

Model = crowd_model(score', 'true_labels',double(groundtruth));
%
Key_MFAB11 = variationalEM_two_coin_crowd_model(Model, 'ell', [2,1,1,1;1,2,1,1;1,1,2,1;1,1,1,2], 'maxIter',100, 'TOL', 1e-3);
prob_error_MFAB11 = mean(Key_MFAB11.ans_labels ~= groundtruth);
error_rate(itr)=(Key_MFAB11.prob_err);

T(end,:,:)=simA(end,:,:);
