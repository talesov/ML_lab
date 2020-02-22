%question 2
clc,clear;
data = load('data.txt');
data_num = length(data);
%store=ones(20);
it=1;
%��ʼѵ�����ݴ�СΪ1000������ľ�Ϊ���Լ�
for num = 500:1000:12000
    % ѵ��������
    training_num = num;
    % test������
    test_num = data_num - training_num;
    %randperm��ǰdatanum�����У����ѡ��test_num��
    test_data_index = randperm(data_num, test_num);
    
    %��ʼ��test��train���ݾ���
    test_data = zeros(test_num, 9);
    training_data = zeros(training_num, 9);
    %������ǩ������
    test_it = 0;
    training_it = 0;

    for i=1:1:data_num
        flag = 0;
        %����data_num��i�ǲ����ڲ���test_data_index������
        for j=1:1:test_num
            if(test_data_index(j)==i)
                flag = 1;
            end
        end
        %�����򽫶�Ӧ�����ݸ��Ƶ�test_data�У������Լ�
        if(flag==1)
            test_it = test_it+1;
            test_data(test_it,:) = data(i,:);
        %���򽫶�Ӧ�����ݸ��Ƶ�traning_data�У���ѵ����
        else
            training_it = training_it+1;
            training_data(training_it,:) = data(i,:);
        end
    end
    %���ò��Ժ���
    [label_pre,sum,success_rate] = LogMLE(test_data,training_data);
  
    store(it)=success_rate;
    x(it)=it;
    it=it+1;
    fprintf('training_num = %d, success_rate = %f, success_num%d\n ',num,success_rate,sum);
end 
plot(x,store,'o-','MarkerFaceColor','r');
