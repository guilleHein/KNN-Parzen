function W_mda= mda_clp( BD,Labels,n_classes)
% mda projection.
% Labels are called 1:n_classes
[N1,N_feat]=size(BD);
Sb=zeros(N_feat,N_feat);
m=mean(BD);
for i_class=1:n_classes;
    index=find(Labels==i_class);
    if length(index)==1
        M_c=BD(index,:);
    else
        M_c=mean(BD(index,:),1);
    end
    Sb=Sb+length(index)*(M_c-m)'*(M_c-m);
end
BD=BD-ones(N1,1)*m;
St=BD'*BD;
Sw=St-Sb;
[V,D]=eig(Sb,Sw);
[~, Index]=sort(abs(diag(D)),'descend');
N_max=min(N_feat,n_classes-1);
Index=Index(1:N_max);
W_mda=V(:,Index);
end


