%ţ�ٷ��߼��ع�
clc;
clear;
x = load('ex2x.dat');
y = load('ex2y.dat');
m = length(x);%��������

xt = x;%����copy
mu = mean(x);%��ֵ
sigma = std(x);%��׼��

%���ݱ�׼��
x = (x - mean(x))./std(x); 
x = [ones(m,1),x] ; %��һ��1��Ϊ�˶�Ӧƫ�ã�û��x���theta��
xt = [ones(m,1),xt] ;

% find����������
pos = find ( y == 1 ); %����
neg = find ( y == 0 ); %������
plot (xt( pos , 2 ),xt( pos , 3 ),'+'); %���Ƶķ���Ҫ�õ�����
hold on;
plot (xt( neg , 2 ),xt( neg , 3 ),'o');
xlabel('Ex_1 score');
ylabel('Ex_2 score');

n = size(x,2);%��������������Ϊ1ʱ��������
H = zeros(n,n);
MaxIter = 1500;%����������
theta = zeros(3,1); %��ʼ�� ��Чtheta zeros(sizeof(x(1,:)))��sizeof����������
e = 1e-6;
alpha = 0.05;%ѧϰ����
g = @(z) 1./(1+exp(-z)); %����sigmoid������@(z)�滻inline

for i = 1:MaxIter
    z = x * theta;
    h = g(z); %�߼��ع�ģ��
    L_theta(i,1) = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));%���������Ȼ����
    delta_L = (1/m)*x'*(h-y); %�����ݶ�  
    %���� Hessian matrix��Ŀ�꺯����theta�Ķ��׵�����
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    %���� L �� theta
    if (i > 1) && (abs(L_theta(i,1) - L_theta(i-1,1)) <= e )
        break;
    end
    theta = theta - H^(-1)*delta_L;
    store(i,:) = [theta',L_theta(i,1)];
end

%�������߽߱磬��Ϊ�����Ǳ�׼����ģ������Ҫ��ԭ��ȥ
x_axis = x(:,2)*sigma(1) + mu(1);
y_axis = (-theta(1,1).*x(:,1) - theta(2,1).*x(:,2))/theta(3,1);
y_axis = y_axis*sigma(2) + mu(2);
plot(x_axis, y_axis,'-');%�ָ���(����)
figure;%�����µĿ�ͼ
plot(1:i-1,store,'-');%���ŵ�����������(theta)�ı仯
legend('\theta_0','\theta_1','\theta_2','L{(\theta)}');
xlabel('iter value');
ylabel('value');

disp("����ʱ��\theta: ");
disp(theta);

x1=20;x2=80;
x1=(x1-mu(1))/sigma(1);
x2=(x2-mu(2))/sigma(2);
theta_x=theta(1)+theta(2)*x1+theta(3)*x2;
disp("����¼ȡ�ĸ��ʣ� ");
disp(g(-theta_x));
