clc,clear
x = load('ex1_2x.dat');
y = load('ex1_2y.dat');
m = length(y);
n = length(x(1,:));
x = [ones(m,1),x];
%���ݱ�׼�������ٵ����������ӿ��ݶ��½��ٶ�(��׼����һ�����������ĵ�ƫ��)
sigma = std(x); % ��׼��
mu = mean(x);   % mu���ʷֲ��ľ�ֵx
x(:,2) = (x(: ,2) - mu(2)) ./ sigma(2); %living area
x(:,3) = (x(: ,3) - mu(3)) ./ sigma(3); %bedroom

theta = zeros(size(x(1,:)))'; %��ʼ��theta 
alpha = [0.05, 0.5, 1,1.2]; %ѧϰ����
J(:) = zeros(50,1);%��ʼ���۾���
for k = 1:4
    theta = zeros(size(x(1,:)))';
    for num_i = 1:50 
        %���ۺ��� �������������h = x * theta -y
        J(k,num_i) = (1/(2*m)) .* (x * theta - y)' * (x * theta - y);
        theta = theta - alpha(k) *(1/ m) * (x' * (x * theta - y)); 
    end
    disp(k);
    disp(theta);
end
figure;
%����J����
plot(0:49,J(1, 1:50),'r-');
hold on;
plot(0:49,J(2, 1:50),'y-');
plot(0:49,J(3, 1:50),'b-');
plot(0:49,J(4, 1:50),'g-');

xlabel ('Number of iterations') 
ylabel ('Cost J ')
legend('alpha=0.05','alpha1=0.5','alpha2=1','alpha3=1.2');

%���theta��alphaΪ0.2ʱ�ģ����ţ�
p = (1650 - mu(2))./sigma(2);
q = (3 - mu(3))./sigma(3);
temp = [1;p;q];
price = theta'*temp;
disp(price);
%ѡ���ѧϰ�ʹ�С�������ٶȻ�Ƚ�����ѧϰ�ʹ����п��ܵ��´��ۺ���������Ž�