clear all
load ./data/nlp_temp.mat;

Model = crowd_model(double(label_matrix'), 'true_labels',double(groundtruth));

% Majority voting:
mv = MajorityVote_crowd_model(Model);

% Dawid & Sknene:
ds = DawidSkene_crowd_model(Model);


% Set parameters:
lambda_worker = 0.25*Model.Ndom^2; lambda_task = lambda_worker * (mean(Model.DegWork)/mean(Model.DegTask)); % regularization parameters
opts={'lambda_worker', lambda_worker, 'lambda_task', lambda_task, 'maxIter',50,'TOL',5*1e-3','verbose',1};
% 1. Categorical minimax entropy:
result1 =  MinimaxEntropy_crowd_model(Model,'algorithm','categorical',opts{:});
% 2. Ordinal minimax entropy:
result2 =  MinimaxEntropy_crowd_model(Model,'algorithm','ordinal',opts{:});



[Nw,Ni]=size(label_matrix);
Nc=max(label_matrix(:));

T=zeros(Nw+1,Ni,Nc);
for s=1:Nc
    T(1:Nw,:,s)=(label_matrix==s);
end
for i=1:Ni
    T(Nw+1,i,ds.ans_labels(i))=1;
end


% R0=[1-1 Nc-1 Nc];
Tinit=T;
for row=2:10 % 9
    for column=ceil(Nc*row/2):Nc*row %max(Nc,ceil(row/Nc)):Nc*row % 8
        T=Tinit;
        num=1;
        R=[row column Nc];
        for itr=1:5
            if num ~= 1
                T(end,:,:)=simA(end,:,:);
            end
            num=num+1;
            
            [B,e]=Tucker_decomposition2(T,R,[50,100,2]);
            simA=B{end};
            for k=1:3
                simA=k_mode_product(simA,B{k},k);
            end
            score=zeros(Nw,Ni);
            for i=1:Nw+1 % Nw
                [~,I]=max(reshape(simA(i,:),[size(simA,2),Nc])');
                score(i,:)=I;
            end
            Model = crowd_model(score', 'true_labels',double(groundtruth));
            mv = MajorityVote_crowd_model(Model);
            error_rate1(row,column,avg,itr)=(mv.error_rate);
            ds = DawidSkene_crowd_model(Model);
            error_rate2(row,column,avg,itr)=(ds.error_rate);
            
        end
    end
end
