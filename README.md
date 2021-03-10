Bioconcentration

clear all
close all
filename=fullfile('Grisoni_et_al_2016_EnvInt88.csv');
data=readtable(filename)

Pre-processing

features=4:12;
features_labels=data.Properties.VariableNames(1,features);
Nfeatures=length(features);
Yclass = data.Class;
class_labels=categorical(unique(Yclass));
Nclassi=length(class_labels);
X = data{:,features};
N=size(X,1);

TrainInd=[];
TestInd=[];
for i=1:N
    if strcmpi (data.Set(i),'Train')
        TrainInd(length(TrainInd)+1)=i;
    else
        TestInd(length(TestInd)+1)=i;
    end
end
Xtest=X(TestInd,:);
Xtrain=X(TrainInd,:);
Xtest=zscore(Xtest);
Xtrain=zscore(Xtrain);
Ytestclass=Yclass(TestInd);
Ytrainclass=Yclass(TrainInd);
primaclassetrain=find(Ytrainclass==1);
secondaclassetrain=find(Ytrainclass==2);
terzaclassetrain=find(Ytrainclass==3);
indiciclassitrain={primaclassetrain secondaclassetrain terzaclassetrain};
primaclassetest=find(Ytestclass==1);
secondaclassetest=find(Ytestclass==2);
terzaclassetest=find(Ytestclass==3);
indiciclassitest={primaclassetest secondaclassetest terzaclassetest};
Ntest=size(Xtest,1);
Ntrain=size(Xtrain,1);
Y_class_ones=zeros(Nclassi,N);
for i=1:Nclassi
    Y_class_ones(i,Yclass==i)=1;
end
Y_class_ones_test=Y_class_ones(:,TestInd);
Y_class_ones_train=Y_class_ones(:,TrainInd);
Visualizzazione dati

[U,Xpc,S]=pca(Xtrain);

figure('position',[100 100 300 200])
bar(1:Nfeatures,100.*S./sum(S))
xlabel('# componenti')
ylabel('% Varianza')
title('Diagramma di Pareto')
box off
figure('Position',[100 100 300 300])
colori='bgr';
for i=1:Nclassi
   line(Xpc(indiciclassitrain{i},1),Xpc(indiciclassitrain{i},2),Xpc(indiciclassitrain{i},3),'color',colori(i),'linestyle','none','marker','o') 
end
axis equal
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
legend(class_labels)
legend boxoff
view(3)

cont=0;
for i=1:Nclassi-1
for j=i+1:Nclassi
    cont=cont+1;
m1=mean(Xtrain(indiciclassitrain{i},:))';
m2=mean(Xtrain(indiciclassitrain{j},:))';
Sw1=cov(Xtrain(indiciclassitrain{i},:))*length(indiciclassitrain{i});
Sw2=cov(Xtrain(indiciclassitrain{j},:))*length(indiciclassitrain{j});
Sw=Sw1+Sw2;
w{cont}=inv(Sw)*(m1-m2);
end
end

Classificatori

NomeClassificatori=["Nb","Mln","SvmL","SvmNl","NN"];
nt=0;

Naive Bayes

nt=nt+1;
Nb=fitcnb(Xtrain,Ytrainclass,'DistributionNames','Normal'); 
[classpredictiontrain{nt,1},PosteriorTrain]=predict(Nb,Xtrain); 
[classpredictiontest{nt,1},PosteriorTest]=predict(Nb,Xtest);
Ytrain_pred{nt}=PosteriorTrain';
Ytest_pred{nt}=PosteriorTest';
[tpr,fpr,~] = roc(Y_class_ones_test,Ytest_pred{nt});
for cl=1:Nclassi
 auc(nt,cl) = sum(tpr{cl}(1:end-1).*diff(fpr{cl}));
end

Multi-Layer Network

nt=nt+1;
Nhidmax=floor(Ntrain/(5*(Nfeatures+Nclassi)));
Numero_partenze=20;
for P=1:Numero_partenze
for Nhid=1:Nhidmax
    net=patternnet(Nhid); 
    neti{Nhid,P}=train(net,Xtrain',Y_class_ones_train);
    net1=neti{Nhid,P};
    Ytest_pred_auc=net1(Xtest');  
    [tpr,fpr,~]=roc(Y_class_ones_test,Ytest_pred_auc);
for cl=1:Nclassi
        auc2(Nhid,cl) = sum(tpr{cl}(1:end-1).*diff(fpr{cl}));
end
end
auc_multiple{P}=auc2;
end

for P=1:Numero_partenze
    auc_medie{P}=mean(auc_multiple{P}');
end

for P=1:Numero_partenze
[massime_auc(P),Numero_neuroni(P)]=max(auc_medie{P});
end

[auc_maggiore,partenza_corretta]=max(massime_auc);
net_tr = neti{Numero_neuroni(partenza_corretta),partenza_corretta};
Ytrain_pred{nt} = net_tr(Xtrain');
Ytest_pred{nt} = net_tr(Xtest');
[~ ,classpredictiontrain{nt,1}]=max(Ytrain_pred{nt});
[~,classpredictiontest{nt,1}]=max(Ytest_pred{nt});
[tpr,fpr,~] = roc(Y_class_ones_test,Ytest_pred{nt});
for cl=1:Nclassi
 auc(nt,cl) = sum(tpr{cl}(1:end-1).*diff(fpr{cl}));
end

Support Vector machines

kernel=["linear","rbf"];
NumeroConfronti=(Nclassi*(Nclassi-1))/2;
for z=1:length(kernel)
    cont=0;
    for i=1:Nclassi-1
    for j=i+1:Nclassi
  indicitrain=sort([indiciclassitrain{i} ; indiciclassitrain{j}]);
  cont=cont+1;
  Xtr=Xtrain(indicitrain,:);
  Ytr=Ytrainclass(indicitrain,:);
 Ytr(Ytr==i)=0;
 Ytr(Ytr==j)=1;
  svm{z,cont} = fitcsvm(Xtr,Ytr,'KernelFunction',kernel(z));
  svm1{z,cont} = fitPosterior(svm{z,cont},'Holdout',0.10);
end
end 
end

for z=1:length(kernel)
choice=zeros(Ntrain,NumeroConfronti);
posterior={};
nt=nt+1;
for i=1:NumeroConfronti
[choice(:,i),posterior{i}] = predict(svm1{z,i},Xtrain);
end
cont=0;
for i=1:NumeroConfronti-1
    for j=i+1:NumeroConfronti
        cont=cont+1;
        A=find(choice(:,cont)==1);
        B=find(choice(:,cont)==0);
        choice(A,cont)=j;
        choice(B,cont)=i;
    end
end    
posterior12=posterior{1};
posterior13=posterior{2};
posterior23=posterior{3};
Denominatore=posterior12(:,1).*posterior13(:,1)+posterior12(:,2).*posterior23(:,1)+posterior13(:,2).*posterior23(:,2);
posterior1=(posterior12(:,1).*posterior13(:,1))./Denominatore;
posterior2=(posterior12(:,2).*posterior23(:,1))./Denominatore;
posterior3=(posterior13(:,2).*posterior23(:,2))./Denominatore;
posteriorsvm=[posterior1 posterior2 posterior3];
Ytrain_pred{nt}=posteriorsvm';
classpredictiontrain{nt,1}=mode(choice');
end

nt=nt-2;
for z=1:length(kernel)
choice=zeros(Ntest,NumeroConfronti);
posterior={};
nt=nt+1;
for i=1:NumeroConfronti
[choice(:,i),posterior{i}] = predict(svm1{z,i},Xtest);
end
cont=0;
for i=1:NumeroConfronti-1
    for j=i+1:NumeroConfronti
        cont=cont+1;
        A=find(choice(:,cont)==1);
        B=find(choice(:,cont)==0);
        choice(A,cont)=j;
        choice(B,cont)=i;
    end
end    
posterior12=posterior{1};
posterior13=posterior{2};
posterior23=posterior{3};
Denominatore=posterior12(:,1).*posterior13(:,1)+posterior12(:,2).*posterior23(:,1)+posterior13(:,2).*posterior23(:,2);
posterior1=(posterior12(:,1).*posterior13(:,1))./Denominatore;
posterior2=(posterior12(:,2).*posterior23(:,1))./Denominatore;
posterior3=(posterior13(:,2).*posterior23(:,2))./Denominatore;
posteriorsvm=[posterior1 posterior2 posterior3];
Ytest_pred{nt}=posteriorsvm';
classpredictiontest{nt,1}=mode(choice');
[tpr,fpr,~] = roc(Y_class_ones_test,Ytest_pred{nt});
for cl=1:Nclassi
 auc(nt,cl) = sum(tpr{cl}(1:end-1).*diff(fpr{cl}));
end
end

K-nearest-neighbours

nt=nt+1;
NN=fitcknn(Xtrain,Ytrainclass,'NumNeighbors',floor(sqrt(Ntrain)));
[classpredictiontrain{nt,1},PosteriorTrain]=predict(NN,Xtrain);
[classpredictiontest{nt,1},PosteriorTest]=predict(NN,Xtest);
Ytrain_pred{nt}=PosteriorTrain';
Ytest_pred{nt}=PosteriorTest';
[tpr,fpr,~] = roc(Y_class_ones_test,Ytest_pred{nt});
for cl=1:Nclassi
 auc(nt,cl) = sum(tpr{cl}(1:end-1).*diff(fpr{cl}));
end
Presentazione dei risultati
Matrici di confusione (Train)
for i = 1 : nt
    figure
plotconfusion(Y_class_ones_train,Ytrain_pred{i})
title(NomeClassificatori{i},"Matrice di confusione (Train)")
set(gcf,'pos',[0 0 350 350])
end
Matrici di confusione (Test)
for i = 1 : nt
    figure
plotconfusion(Y_class_ones_test,Ytest_pred{i})
title(NomeClassificatori{i},"Matrice di confusione (Test)")
set(gcf,'pos',[0 0 350 350])
end
Curve ROC
for i = 1 : nt
    figure
plotroc(Y_class_ones_test,Ytest_pred{i})
title(NomeClassificatori{i},"Curve ROC")
set(gcf,'pos',[0 0 330 330])
end
AUC delle singole classi
figure
bar(auc)
box off
xticklabels(NomeClassificatori)
legend(class_labels)
legend boxoff
title('AUC')
    AUC Mediata sulle tre classi
figure
bar(mean(auc'))
box off
xticklabels(NomeClassificatori)
title('AUC media')
Classe predetta da ogni classificatore
nt=nt+1;
classpredictiontrain{nt,1}=Ytrainclass;
classpredictiontest{nt,1}=Ytestclass;
for i=1:nt
    classpredictiontest{i}=reshape(classpredictiontest{i},1,Ntest);
    classpredictiontrain{i}=reshape(classpredictiontrain{i},1,Ntrain);
end
name=[NomeClassificatori "Classe esatta"]';
Test=cell2table(classpredictiontest,'RowNames',name)
Train=cell2table(classpredictiontrain,'RowNames',name)
Numero di neuroni dello strato nascosto utilizzati
fprintf('il numero di neuroni dello strato nascosto utilizzato è %d .\n',Numero_neuroni(partenza_corretta))
Feature più importanti per le discriminazioni tra classi
cont=0;
for i=1:Nclassi-1
    for j=i+1:Nclassi
       cont=cont+1; 
        figure('Position',[0 0 800 800])
       bar(abs(w{cont}))
       box off
       classe1confronto=string(i);
       classe2confronto=string(j);
       titolo=strcat("Discriminatore di Fischer tra classe ",classe1confronto," e ",classe2confronto);
       title(titolo)
       ylabel('ampiezza vettore W')
       xticklabels(features_labels)
    end
end
