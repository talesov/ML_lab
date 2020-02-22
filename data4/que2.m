%question 2
clc,clear;
data = load('data.txt');
data_num = length(data);
%store=ones(20);
it=1;
%起始训练数据大小为1000，其余的均为测试集
for num = 500:1000:12000
    % 训练数据量
    training_num = num;
    % test数据量
    test_num = data_num - training_num;
    %randperm将前datanum个数中，随机选择test_num个
    test_data_index = randperm(data_num, test_num);
    
    %初始化test和train数据矩阵
    test_data = zeros(test_num, 9);
    training_data = zeros(training_num, 9);
    %两个标签计数器
    test_it = 0;
    training_it = 0;

    for i=1:1:data_num
        flag = 0;
        %查找data_num的i是不是在不在test_data_index数组中
        for j=1:1:test_num
            if(test_data_index(j)==i)
                flag = 1;
            end
        end
        %若是则将对应的数据复制到test_data中，即测试集
        if(flag==1)
            test_it = test_it+1;
            test_data(test_it,:) = data(i,:);
        %否则将对应的数据复制到traning_data中，即训练集
        else
            training_it = training_it+1;
            training_data(training_it,:) = data(i,:);
        end
    end
    %调用测试函数
    [label_pre,sum,success_rate] = LogMLE(test_data,training_data);
  
    store(it)=success_rate;
    x(it)=it;
    it=it+1;
    fprintf('training_num = %d, success_rate = %f, success_num%d\n ',num,success_rate,sum);
end 
plot(x,store,'o-','MarkerFaceColor','r');
