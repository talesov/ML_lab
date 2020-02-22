%手写识别第二问，对不同C进行计算
clc,clear;
train_data = load('hand_digits_train.dat');
test_data = load('hand_digits_test.dat');

trainnum=3000;%训练数据量
train_len = size(train_data,1); 
test_len = size(test_data,1); 
train_index = randperm(train_len,trainnum);
test_index = randperm(test_len,2115);
train_select = zeros(trainnum,785);
test_select = zeros(2115,785);

for i = 1:trainnum
    train_select(i,:) = train_data(train_index(i),:);
end
for i = 1:2115
    test_select(i,:) = test_data(test_index(i),:);
end

C = [0.00001,0.1,1,10,10000];
%有正则项
for i=1:size(C,2)
    predict2(train_select,test_select,'linear',0,C(i));
end
