%�����߼��ع�
clc,clear;
x=load("ex3Logx.dat");%����������ڴ��ļ�
y=load("ex3Logy.dat");%���ļ�Ϊ����
%figure
pos=find(y==1);%���ط���1��������
neg=find(y==0);
%plot(x(pos,1),x(pos,2),'+');%x�еĵ�һ��Ϊ�����꣬�ڶ���Ϊ������
%hold on
%plot(x(neg,1),x(neg,2),'o');

%��������x
u=x(:,1);
v=x(:,2);
xx=map_feature(u,v);%����������6�������һ����28��
[m,n]=size(xx);
%t=length(xx(1,:)); %����ʽn��Ч��������
%ţ�ٷ�
lambda=[0,1,10];%�������
g=@(z) 1.0 ./ (1+exp(-z)); %g���� sigmoid����(S�ͺ��� ����ڷֶκ�����������)

L=eye(n,n);%�ԽǾ���
L(1,1)=0;%��ʼλ����(1,1)��Ϊ0
itera=15;
%���е���
for k=1:length(lambda)%ѭ��ÿһ�β�ͬ��lambda(����ϵ��)
    theta=zeros(n,1);%��ÿ�β�ͬ��lambda��ϵ��theta���г�ʼ��
    for i=1:itera
        %����ع�ģ��
        z=xx*theta;
        h=g(z);
        %���ۺ���
        J=-(1/m)*sum(y.*log(h)+(1-y).*log(1-h))+(lambda(1,k)/(2*m))*sum(theta(2:end).^2);
        %��ɭ����ţ�ٷ���
        H=(1/m).*xx'*diag(h)*diag(1-h)*xx+(lambda(1,k)/m)*L;
        %�����ݶ�
        T=(lambda(1,k)/m).*theta;
        T(1,1)=0;%��һ��theta����������
        delta_J=(1/m).*xx'*(h-y)+T;
        %��������theta����ɭ�������棩
        theta=theta-H^(-1)*delta_J;
        store(i,k)=J;
        disp(J);
        %norm(theta);
    end
    %��������
    u=linspace(-1,1.5,200);
    v=linspace(-1,1.5,200);
    t=length(u);
    b=zeros(t,t);
    %����z
    for i=1:t
        for j=1:t
            b(i,j)=map_feature(u(i),v(j))*theta;%6��degree
        end
    end
    figure;
    plot(x(pos,1),x(pos,2),'+');%x�еĵ�һ��Ϊ�����꣬�ڶ���Ϊ������
    hold on
    plot(x(neg,1),x(neg,2),'o');

    b = b';
    contour(u, v, b, [0, 0], 'LineWidth', 2);
    legend('y = 1', 'y = 0', 'Decision boundary');
    title(sprintf('\\lambda = %g', lambda(1,k)), 'FontSize', 10);
    fprintf('norm=%g',norm(theta));
    %disp(norm(theta));
end
%���ƴ��ۺ���J���������
for i=1:length(lambda)
    figure;
    plot(1:itera,store(:,i),'o-','MarkerFaceColor','r');
    title(sprintf('\\lambda=%g',lambda(1,i)),'FontSize',10);
    legend('J(\theta)');
end
