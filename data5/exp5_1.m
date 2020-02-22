%lab5_1 第一问和第二问
clc,clear;
type='linear';
C=1;%正则化项

%第四个参数为gamma，默认为0
%参数：train_data_name,test_data_name,kertype,gamma,C
predict1('training_1.txt','test_1.txt',type,0,C);
predict1('training_2.txt','test_2.txt',type,0,C);