clc, clear;
x = load('ex1_1x.dat');
y = load('ex1_1y.dat');

ylabel('Height in meters');
xlabel('Age in years');
m = length(y);
x = [ones(m, 1), x]; 
alpha = 0.07; %ѧϰ����
itera = 1500; %��������
theta=zeros(2,1); %theta����
for i = 1:itera %��theta���е���
	theta(1) = theta(1) - alpha *( 1/m ) * sum(x * theta - y); 
	theta(2) = theta(2) - alpha *( 1/m ) * sum((x * theta - y) .* x(:, 2));
end 
plot(x(:, 2), y, 'o');%����ԭ���ݵ�
hold on  %����
plot(x(:, 2), x * theta, '-') %����Ԥ������
legend('Training data', 'Linear regression')

age = [1 3.5;1 7];%Ԥ����ֵ
height = age * theta;
disp(height);

%J(theta)
J_vals = zeros(100 ,100); %��ʼ��0����
theta0_vals = linspace(-3 ,3 ,100);
theta1_vals = linspace(-1 ,1 ,100);
%����linespace(x1,x2,N)������x1��x2��N�ֱ�Ϊ��ʼֵ����ֵֹ��Ԫ�ظ�����
for i = 1:length(theta0_vals)
	for j = 1:length(theta1_vals)
        t = [theta0_vals(i);theta1_vals(j)];%��Ӧtheta��
        %�����ת�á�������ƽ��
    	J_vals(i, j) = (1 / (2*m)) .* (x * t - y)' * (x * t - y);
	end
end
J_vals = J_vals'; 
figure; 
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel ('\theta_1') 