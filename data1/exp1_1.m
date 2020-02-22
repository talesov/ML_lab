clc, clear;
x = load('ex1_1x.dat');
y = load('ex1_1y.dat');

ylabel('Height in meters');
xlabel('Age in years');
m = length(y);
x = [ones(m, 1), x]; 
alpha = 0.07; %学习因子
itera = 1500; %迭代次数
theta=zeros(2,1); %theta矩阵
for i = 1:itera %对theta进行迭代
	theta(1) = theta(1) - alpha *( 1/m ) * sum(x * theta - y); 
	theta(2) = theta(2) - alpha *( 1/m ) * sum((x * theta - y) .* x(:, 2));
end 
plot(x(:, 2), y, 'o');%绘制原数据点
hold on  %保持
plot(x(:, 2), x * theta, '-') %绘制预测曲线
legend('Training data', 'Linear regression')

age = [1 3.5;1 7];%预测数值
height = age * theta;
disp(height);

%J(theta)
J_vals = zeros(100 ,100); %初始化0矩阵
theta0_vals = linspace(-3 ,3 ,100);
theta1_vals = linspace(-1 ,1 ,100);
%对于linespace(x1,x2,N)，其中x1、x2、N分别为起始值、终止值、元素个数。
for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
        t = [theta0_vals(i);theta1_vals(j)];%对应theta对
        %矩阵的转置×矩阵是平方
    	J_vals(i, j) = (1 / (2*m)) .* (x * t - y)' * (x * t - y);
	end
end
J_vals = J_vals'; 
figure; 
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel ('\theta_1') 