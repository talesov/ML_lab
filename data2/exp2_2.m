%牛顿法逻辑回归
clc;
clear;
x = load('ex2x.dat');
y = load('ex2y.dat');
m = length(x);%样本长度

xt = x;%样本copy
mu = mean(x);%均值
sigma = std(x);%标准差

%数据标准化
x = (x - mean(x))./std(x); 
x = [ones(m,1),x] ; %加一列1，为了对应偏置（没有x项的theta）
xt = [ones(m,1),xt] ;

% find返回行索引
pos = find ( y == 1 ); %及格
neg = find ( y == 0 ); %不及格
plot (xt( pos , 2 ),xt( pos , 3 ),'+'); %绘制的符号要用单引号
hold on;
plot (xt( neg , 2 ),xt( neg , 3 ),'o');
xlabel('Ex_1 score');
ylabel('Ex_2 score');

n = size(x,2);%返回列数，参数为1时返回行数
H = zeros(n,n);
MaxIter = 1500;%设置最大迭代
theta = zeros(3,1); %初始化 等效theta zeros(sizeof(x(1,:)))，sizeof返回行列数
e = 1e-6;
alpha = 0.05;%学习因子
g = @(z) 1./(1+exp(-z)); %定义sigmoid函数，@(z)替换inline

for i = 1:MaxIter
    z = x * theta;
    h = g(z); %逻辑回归模型
    L_theta(i,1) = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));%极大对数似然函数
    delta_L = (1/m)*x'*(h-y); %计算梯度  
    %计算 Hessian matrix（目标函数对theta的二阶导数）
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    %更新 L 和 theta
    if (i > 1) && (abs(L_theta(i,1) - L_theta(i-1,1)) <= e )
        break;
    end
    theta = theta - H^(-1)*delta_L;
    store(i,:) = [theta',L_theta(i,1)];
end

%画出决策边界，因为数据是标准化后的，因此需要还原回去
x_axis = x(:,2)*sigma(1) + mu(1);
y_axis = (-theta(1,1).*x(:,1) - theta(2,1).*x(:,2))/theta(3,1);
y_axis = y_axis*sigma(2) + mu(2);
plot(x_axis, y_axis,'-');%分割线(分类)
figure;%创建新的框图
plot(1:i-1,store,'-');%随着迭代，各因数(theta)的变化
legend('\theta_0','\theta_1','\theta_2','L{(\theta)}');
xlabel('iter value');
ylabel('value');

disp("收敛时的\theta: ");
disp(theta);

x1=20;x2=80;
x1=(x1-mu(1))/sigma(1);
x2=(x2-mu(2))/sigma(2);
theta_x=theta(1)+theta(2)*x1+theta(3)*x2;
disp("不被录取的概率： ");
disp(g(-theta_x));
