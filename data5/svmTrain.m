
function svm=svmTrain(data,label,kertype,gamma,C)
%minimizes 1/2*x'*H*x+f'*x
    options=optimset;% Options是用来控制算法的选项参数的struct
    options.LargerScale='off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
    options.Display='off';%这样设置意味着没有输出
    
    n=length(label);%数组Y的长度
    H=(label*label').*kernel(data,data,kertype,gamma);%调用kernel函数，
    f=-ones(n,1);%f为1*n个-1,f相当于Quadprog函数中的c
    A=[];
    b=[];
    Aeq=label';%相当于Quadprog函数中的A1,b1
    beq=0;
    lb=zeros(n,1);%相当于Quadprog函数中的LB，UB
%     ub=C*ones(n,1);
    if C == 0  %无正则项
        ub = [];
    else       %有正则项
        ub = C.*ones(n,1);
    end
    x0=zeros(n,1);% x0是解的初始近似值
    [a,fval,exitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    
    eps=1e-7;%阈值选择
    
    %寻找支持向量
    svm_label=find(abs(a)>eps);%0<a<a(max)则认为x为支持向量     
    svm.alpha=a(svm_label);%拉格朗日乘子向量
    svm.Xsv=data(svm_label,:);%支持向量坐标向量
    svm.Ysv=label(svm_label);%支持向量label
    svm.len=length(svm_label);%支持向量个数
    svm.a=a;
    
end