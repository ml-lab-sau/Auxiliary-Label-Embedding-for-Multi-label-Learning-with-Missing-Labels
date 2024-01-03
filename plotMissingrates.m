x=readtable('D:\Nadira\data\Parameter Sensitivity\Missing rate 0.3\para\AUCv.csv');
%x=readtable('D:\Sayed Mortaza\Reports\Ranking-Evaluation\Missing0.10.30.6\AUC.csv');
%gEtting data from tree file
% d1=readtable('D:\Sayed Mortaza\NFML-Reports\FeatureNoise\All-FeatureNoise-0-Missing0.3.csv');
% d2=readtable('D:\Sayed Mortaza\NFML-Reports\FeatureNoise\All-FeatureNoise-0.3-Mising0.3.csv');
% d3=readtable('D:\Sayed Mortaza\NFML-Reports\FeatureNoise\All-FeatureNoise-0.5-Missing0.3.csv');
% d1=table2array(d1);
% d2=table2array(d2);
% d3=table2array(d3);
% x(1,:)=d1(25,:);
% x(2,:)=d2(25,:);
% x(3,:)=d3(25,:);
% 
 y=[-5 -4 -3 -2 -1 0 1 2 3 ];
% whichData=1;
% for i=1:2
%     x1(i)=x(whichData,1);
%     x2(i)=x(whichData,2);
%     x3(i)=x(whichData,3);
%     x4(i)=x(whichData,4);
%     x5(i)=x(whichData,5);
%     x6(i)=x(whichData,6);
%     x7(i)=x(whichData,7);
%     x8(i)=x(whichData,8);
%     whichData=whichData+1;
% end
data=table2array(x);
data=data';
lambda1=data(1,:);
lambda2=data(2,:);
lambda3=data(3,:);
lambda4=data(4,:);
gamma=data(5,:);
eta=data(6,:);

figure
plot(y,lambda1,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5]);
hold on
plot(y,lambda2,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5]);
hold on
plot(y,lambda3,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5]);
hold on
plot(y,lambda4,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5])
hold on
plot(y,gamma,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5])

hold on
plot(y,eta,'-', 'LineWidth',1,'MarkerSize',2,'MarkerFaceColor',[0.4,0.5,0.5])
title('Parameter Sensetivity for Medical Dataset');
ylabel('Ranking Loss (mean) ');
xlabel("Paramter's Values");
legend('\lambda_1','\lambda_2','\lambda_3','\lambda_4','\gamma','\eta')
