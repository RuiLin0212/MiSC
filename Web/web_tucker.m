clear all
load web.mat;
I=find(groundtruth~=0);
label_matrix=label_matrix(:,I);
groundtruth=groundtruth(I);

Model = crowd_model(double(label_matrix'), 'true_labels',double(groundtruth));


% Majority voting:
mv = MajorityVote_crowd_model(Model);

% Dawid & Sknene:
ds = DawidSkene_crowd_model(Model);


[Nw,Ni]=size(label_matrix);
Nc=double(max(label_matrix(:)));
T=zeros(Nw+1,Ni,Nc);
for s=1:Nc
    T(1:Nw,:,s)=(label_matrix==s);
end
for i=1:Ni
    T(Nw+1,i,ds.ans_labels(i))=1;
end

Tinit=T;
for row=3:10 
    for column=ceil(Nc*row/2):Nc*row
        for avg=1:10
            T=Tinit;
            temp1=T(1:Nw,:,:);
            temp2=T(end,:,:);
            I=find(temp1==1);
            pp=randperm(length(I));
            pp=pp(1:ceil(length(I)*0.9));
            I=I(pp);
            T=zeros(size(temp1));
            T(I)=1;
            T=[T;temp2];
            num=1;
            R=[row column Nc];
            for itr=1:5
                if num ~= 1
                    T(end,:,:)=simA(end,:,:);
                end
                num=num+1;

                [B,e]=Tucker_decomposition2(T,R,[100,500,5]);
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
                error_rate1(row,column,itr)=(mv.error_rate);
                ds = DawidSkene_crowd_model(Model);
                error_rate2(row,column,itr)=(ds.error_rate);
                
                reshape(simA(end,:,:),[Ni,Nc]);
                [~,I]=max(ans');
                error_rate3(row,column,itr)=length(find(I~=groundtruth))/Ni*100;
            end
            
        end
    end
end
