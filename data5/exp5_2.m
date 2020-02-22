%手写识别的主函数
clc,clear;
train_data = load('hand_digits_train.dat');
test_data = load('hand_digits_test.dat');

trainnum=3000;

train_len = size(train_data,1); 
test_len = size(test_data,1); 
train_index = randperm(train_len,trainnum);
test_index = randperm(test_len,2115);
train_select = zeros(trainnum,785);
test_select = zeros(2115,785);

%选取部分数据进行训练
for i = 1:trainnum
    train_select(i,:) = train_data(train_index(i),:);
end
for i = 1:2115
    test_select(i,:) = test_data(test_index(i),:);
end

%无正则项 
[train_miss_index,test_miss_index] = predict2(train_select,test_select,'linear',0,0);
%查看被错误分类的手写字体
%训练错误手写字体
for i = 1:length(train_miss_index)
   strimage('train-01-images.svm',i);
   figure;%生成一个新的figure，防止覆盖图像展示
end
%测试错误手写字体
for i = 1:length(test_miss_index)
   strimage('test-01-images.svm',i);
   if i~=length(test_miss_index)
       figure;
   end
end
