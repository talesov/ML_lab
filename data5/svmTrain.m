
function svm=svmTrain(data,label,kertype,gamma,C)
%minimizes 1/2*x'*H*x+f'*x
    options=optimset;% Options�����������㷨��ѡ�������struct
    options.LargerScale='off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
    options.Display='off';%����������ζ��û�����
    
    n=length(label);%����Y�ĳ���
    H=(label*label').*kernel(data,data,kertype,gamma);%����kernel������
    f=-ones(n,1);%fΪ1*n��-1,f�൱��Quadprog�����е�c
    A=[];
    b=[];
    Aeq=label';%�൱��Quadprog�����е�A1,b1
    beq=0;
    lb=zeros(n,1);%�൱��Quadprog�����е�LB��UB
%     ub=C*ones(n,1);
    if C == 0  %��������
        ub = [];
    else       %��������
        ub = C.*ones(n,1);
    end
    x0=zeros(n,1);% x0�ǽ�ĳ�ʼ����ֵ
    [a,fval,exitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    
    eps=1e-7;%��ֵѡ��
    
    %Ѱ��֧������
    svm_label=find(abs(a)>eps);%0<a<a(max)����ΪxΪ֧������     
    svm.alpha=a(svm_label);%�������ճ�������
    svm.Xsv=data(svm_label,:);%֧��������������
    svm.Ysv=label(svm_label);%֧������label
    svm.len=length(svm_label);%֧����������
    svm.a=a;
    
end