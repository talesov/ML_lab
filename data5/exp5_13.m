%lab5_1 ������ �ı�C�Ĵ�С
clc,clear;
type='linear';
C=[0.1,1,10,100];

for i=1:size(C,2)
    predict1('training_1.txt','test_1.txt',type,0,C(i));
end

for i=1:size(C,2)
    predict1('training_2.txt','test_2.txt',type,0,C(i));
end