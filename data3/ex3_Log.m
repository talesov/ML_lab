%正则化逻辑回归
clc,clear;
x=load("ex3Logx.dat");%横纵坐标均在此文件
y=load("ex3Logy.dat");%此文件为分类
%figure
pos=find(y==1);%返回分类1的行索引
neg=find(y==0);
%plot(x(pos,1),x(pos,2),'+');%x中的第一列为横坐标，第二列为纵坐标
%hold on
%plot(x(neg,1),x(neg,2),'o');

%特征向量x
u=x(:,1);
v=x(:,2);
xx=map_feature(u,v);%增加特征至6次项，特征一共有28个
[m,n]=size(xx);
%t=length(xx(1,:)); %与上式n等效，求列数
%牛顿法
lambda=[0,1,10];%正则参数
g=@(z) 1.0 ./ (1+exp(-z)); %g函数 sigmoid函数(S型函数 相较于分段函数有连续性)

L=eye(n,n);%对角矩阵
L(1,1)=0;%起始位置是(1,1)置为0
itera=15;
%进行迭代
for k=1:length(lambda)%循环每一次不同的lambda(正则系数)
    theta=zeros(n,1);%对每次不同的lambda的系数theta进行初始化
    for i=1:itera
        %计算回归模型
        z=xx*theta;
        h=g(z);
        %代价函数
        J=-(1/m)*sum(y.*log(h)+(1-y).*log(1-h))+(lambda(1,k)/(2*m))*sum(theta(2:end).^2);
        %海森矩阵（牛顿法）
        H=(1/m).*xx'*diag(h)*diag(1-h)*xx+(lambda(1,k)/m)*L;
        %计算梯度
        T=(lambda(1,k)/m).*theta;
        T(1,1)=0;%第一个theta不计算正则
        delta_J=(1/m).*xx'*(h-y)+T;
        %迭代更新theta（海森矩阵求逆）
        theta=theta-H^(-1)*delta_J;
        store(i,k)=J;
        disp(J);
        %norm(theta);
    end
    %生成数据
    u=linspace(-1,1.5,200);
    v=linspace(-1,1.5,200);
    t=length(u);
    b=zeros(t,t);
    %评估z
    for i=1:t
        for j=1:t
            b(i,j)=map_feature(u(i),v(j))*theta;%6是degree
        end
    end
    figure;
    plot(x(pos,1),x(pos,2),'+');%x中的第一列为横坐标，第二列为纵坐标
    hold on
    plot(x(neg,1),x(neg,2),'o');

    b = b';
    contour(u, v, b, [0, 0], 'LineWidth', 2);
    legend('y = 1', 'y = 0', 'Decision boundary');
    title(sprintf('\\lambda = %g', lambda(1,k)), 'FontSize', 10);
    fprintf('norm=%g',norm(theta));
    %disp(norm(theta));
end
%绘制代价函数J的收敛情况
for i=1:length(lambda)
    figure;
    plot(1:itera,store(:,i),'o-','MarkerFaceColor','r');
    title(sprintf('\\lambda=%g',lambda(1,i)),'FontSize',10);
    legend('J(\theta)');
end
