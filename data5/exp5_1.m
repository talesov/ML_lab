%lab5_1 ��һ�ʺ͵ڶ���
clc,clear;
type='linear';
C=1;%������

%���ĸ�����Ϊgamma��Ĭ��Ϊ0
%������train_data_name,test_data_name,kertype,gamma,C
predict1('training_1.txt','test_1.txt',type,0,C);
predict1('training_2.txt','test_2.txt',type,0,C);