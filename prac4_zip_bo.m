% updated to Matlab2015
% MC 2016
close all;
clear;
clc;
%OPCIONES
i_dib=0;					%0 NO /1 SI: DIBUJOS DE DIGITOS
i_CM=0;						%0 NO /1 SI: CALCULA MATRIZ DE CONFUSION
N_classes=10;
                     %PARAMETRO K en knn
%% Elección de la transformada y reducción de dimensión
disp(' ')
disp('Elegir Transformada')
i_transform=input(' No transformar (0), DCT (1)  Hadamard (2) = ');
if i_transform >0
    disp(' ')
    disp('Elegir Dimensión Reducida')
    N_dim=input(' Dim =  ');
else
    N_dim=16;
end
N_feat=N_dim*N_dim;
for part = 1:10
%% Lectura BD de train
X_train=[];             % Matriz de Nx256 que contiene todos los vectores
                        % Cada muestra va entre 0(Negro) y 1(Blanco)
Labels_train=[];        % Etiquetas (Inicialmente los datos estan ordenados
                        % por clases del 0 al 9)
for k=0:N_classes-1
    nombre=sprintf('train%d.txt',k);  
    [data] = textread(nombre,'','delimiter',',');
    %data=round(data);  %OPCIONAL elimina los grises
                            %y lo deja todo en blanco y negro
    X_train=[X_train;data];
    N_size=size(data);
    Labels_train=[Labels_train;k*ones(N_size(1),1)];
end
%clear nombre data N_size k

%% Lectura BD de test
nombre=sprintf('zip.test');
[data] = textread(nombre,'','delimiter',' ');
Labels_test =data(:,1);
X_test=data(:,2:size(data,2));
%clear nombre data


%% stratified data partitioning
X=[X_train; X_test];
Labels=[Labels_train; Labels_test];
P_train=0.5;  % Train and test sizes are equal
Index_train=[];
Index_test=[];
for i_class=0:N_classes-1
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_test] = dividerand(N_i_class,P_train,1-P_train);
    Index_train=[Index_train;index(I_train)];
    Index_test=[Index_test;index(I_test)];
end
% Train Selection and mixing
for k_neig=1:10 
X_train=X(Index_train,:);
Labels_train=Labels(Index_train);
% Test Selection and mixing
X_test=X(Index_test,:);
Labels_test=Labels(Index_test);
%clear Index_train Index_test index i_class N_i_class I_train I_test


%% OPCION TRANSFORMADAS
A2=hadamard(N_dim);
if i_transform >0
    % Transformamos BD de train
    A=hadamard(16);
    N_d2=floor(16/N_dim);
    N_samples=size(X_train,1);
    X_aux=zeros(size(X_train,1),N_feat);
    if i_transform==1
        % DCT
        for i_samples=1:N_samples
            data=X_train(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=dct2(data);
            data=data(1:N_dim,1:N_dim);
            X_aux(i_samples,:)=data(:)';
        end
    else
        % Hadamard
        for i_samples=1:N_samples
            data=X_train(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=A*data*A';
            data=data(1:N_d2:16,1:N_d2:16);
            X_aux(i_samples,:)=data(:)';
        end
    end
    
    if i_dib==1
        figure('name','Dominio Transformado')
        for k=0:N_classes-1
            subplot(3,4,k+1)
            ind=find(Labels_train==k);
            N_ale=randi(length(ind));
            data=X_aux(ind(N_ale),:);
            data=reshape(data,N_dim,N_dim);
            imagesc(abs(data));
            colorbar
            xlabel(k)
        end
        %clear N_ale ind k data
    end
    X_train=X_aux;
    
    %Transformamos BD de test
    N_samples=size(X_test,1);
    X_aux=zeros(size(X_test,1),N_feat);
    if i_transform==1
        % DCT
        for i_samples=1:N_samples
            data=X_test(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=dct2(data);
            data=data(1:N_dim,1:N_dim);
            X_aux(i_samples,:)=data(:)';
        end
    else
        % Hadamard
        for i_samples=1:N_samples
            data=X_test(i_samples,:);
            data=reshape(data,16,16);
            data=data';
            data=A*data*A';
            data=data(1:N_d2:16,1:N_d2:16);
            X_aux(i_samples,:)=data(:)';
        end
    end
    X_test=X_aux;
    %clear X_aux N_samples A i_samples N_d2
end

%% OPCION dibujos de imagenes

%clear i_dib A2 

%% Create a default (linear) discriminant analysis classifier:
%linclass = fitcdiscr(X_train,Labels_train,'prior','empirical');
%Linear_out = predict(linclass,X_train);
%Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
%fprintf(1,' error Linear train = %g   \n', Linear_Pe_train)
%Linear_out = predict(linclass,X_test);
%Linear_Pe_test[k_neig,part]=sum(Labels_test ~= Linear_out)/length(Labels_test);
%fprintf(1,' error Linear test = %g   \n', Linear_Pe_test)
% Test confusion matrix
%if i_CM==1
 %   CM_Linear_test=confusionmat(Labels_test,Linear_out)
%end

%% Create a knn classifier:
knnclass = fitcknn(X_train,Labels_train,'NumNeighbors',k_neig);
knn_out = predict(knnclass,X_train);
knn_Pe_train(k_neig,part)=sum(Labels_train ~= knn_out)/length(Labels_train);
fprintf(1,' error knn train = %g   \n', knn_Pe_train)
knn_out = predict(knnclass,X_test);
knn_Pe_test(k_neig,part)=sum(Labels_test ~= knn_out)/length(Labels_test);
fprintf(1,' error knn test = %g   \n', knn_Pe_test)
% Test confusion matrix
if i_CM==1
    CM_knn_test=confusionmat(Labels_test,knn_out)
end
end
end
for i=1:10
    err_knn_train(i)=mean(knn_Pe_train(i,:));
    err_knn_test(i)=mean(knn_Pe_test(i,:));
end
figure
title('validacion parametro k en KNN')
plot(err_knn_train);
hold on
plot(err_knn_test);
legend('train','test')
xlabel('K')
ylabel('Pe')
hold off
