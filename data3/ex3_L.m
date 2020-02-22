%�������Իع�
clc,clear;
x=load("ex3Linx.dat");
y=load("ex3Liny.dat");
plot(x,y,'o','MarkerFaceColor','r');
hold on;
%legend('training data');

m=length(x);
x=[ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
lambda=[0;1;10];%���򻯲���
L=diag([0,1,1,1,1,1]);%�ԽǾ���
%theta=zeros(6,3);
xt=linspace(-1,1,100)';%����x����
n=length(xt);
xx=[ones(n,1),xt,xt.^2,xt.^3,xt.^4,xt.^5];
%���淽����⣨��С���˷���
for i=1:length(lambda)
    theta = ((x' * x + lambda(i) * L)^-1) * x' * y; 
    plot(xt,xx*theta,'--');
    hold on;
end
legend('training data','\lambda=0','\lambda=1','\lambda=10');



