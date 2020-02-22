clc,clear
x = load('ex1_2x.dat');
y = load('ex1_2y.dat');
m = length(y);
n = length(x(1,:));
x = [ones(m,1),x];
%数据标准化，减少迭代次数，加快梯度下降速度(标准化归一，数据离中心的偏差)
sigma = std(x); % 标准差
mu = mean(x);   % mu概率分布的均值x
x(:,2) = (x(: ,2) - mu(2)) ./ sigma(2); %living area
x(:,3) = (x(: ,3) - mu(3)) ./ sigma(3); %bedroom

theta = zeros(size(x(1,:)))'; %初始化theta 
alpha = [0.05, 0.5, 1,1.2]; %学习因子
J(:) = zeros(50,1);%初始代价矩阵
for k = 1:4
    theta = zeros(size(x(1,:)))';
    for num_i = 1:50 
        %代价函数 列向量计算出来h = x * theta -y
        J(k,num_i) = (1/(2*m)) .* (x * theta - y)' * (x * theta - y);
        theta = theta - alpha(k) *(1/ m) * (x' * (x * theta - y)); 
    end
    disp(k);
    disp(theta);
end
figure;
%进行J绘制
plot(0:49,J(1, 1:50),'r-');
hold on;
plot(0:49,J(2, 1:50),'y-');
plot(0:49,J(3, 1:50),'b-');
plot(0:49,J(4, 1:50),'g-');

xlabel ('Number of iterations') 
ylabel ('Cost J ')
legend('alpha=0.05','alpha1=0.5','alpha2=1','alpha3=1.2');

%最后theta是alpha为0.2时的，缩放：
p = (1650 - mu(2))./sigma(2);
q = (3 - mu(3))./sigma(3);
temp = [1;p;q];
price = theta'*temp;
disp(price);
%选择的学习率过小，收敛速度会比较慢。学习率过大，有可能导致代价函数错过最优解