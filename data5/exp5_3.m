%第三题非线性SVM主函数
clc,clear;
type = 'rbf'; %使用rbf合函数
train_name = load('training_3.text');
gamma = [1,10,100,1000];
C = 1;
for i = 1:length(gamma)
   predict3(train_name,type,gamma(i),C); 
   if i ~= length(gamma)%不等于
       figure;
   end
end
