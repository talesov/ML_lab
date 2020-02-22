%问题1
%实验所给的convData.m是将大小为12960的nursery.data.txt数据集
%分割成一份大小为10000的训练集training_data.txt
%和大小为2960的测试集test_data.txt
clc,clear;
train = load('training_data.txt');
test = load('test_data.txt');

[train_row,train_col] = size(train);
test_num = size(test,1);
%记录每个类对应的对数似然函数值，最大值对应的下标减1即最终我们预测的类
label_for_MLE = zeros(test_num,5);
label_pre = zeros(test_num,1); %存储所有我们预测的类
label_num = max(train(:,end)); %类别总数
for i = 1:test_num%（1：2960）不同测试样本，即每一行
    for y = 0:label_num%（0：4）不同类别
        count_y = length(find(train(:,train_col)==y));%每一个类别的个数
        p_y = count_y/train_row;%每一个类别/总数
        log_count_xy = 0;
        for j = 1:train_col-1%（1：8）不同feature，即sample每一列
            %类别为y条件下，抽中对应test.featurej在train.featurej的概率
            count_xy = length(find(train(:,train_col)==y & train(:,j)==test(i,j)));
            %拉普拉斯平滑（根据不同的feature）类条件概率，y为类别
            if j==1
                p_xy = (count_xy+1)/(count_y+3);%分子+1，分母+feature类别数
            elseif j==2
                p_xy = (count_xy+1)/(count_y+5);
            elseif j==3
                p_xy = (count_xy+1)/(count_y+4);
            elseif j==4
                p_xy = (count_xy+1)/(count_y+4);
            elseif j==5
                p_xy = (count_xy+1)/(count_y+3);
            elseif j==6
                p_xy = (count_xy+1)/(count_y+2);
            elseif j==7
                p_xy = (count_xy+1)/(count_y+3);
            elseif j==8
                p_xy = (count_xy+1)/(count_y+3);
            end
            log_count_xy = log_count_xy + log(p_xy);%乘法转换为加法，
            %条件概率根据独立性连乘转化为连加
        end
        %对每一个test，按列存储每一个类别的概率值，贝叶斯分子，因为分母相同只考虑分子
        label_for_MLE(i,y+1) = log(p_y) + log_count_xy;%py类先验概率，xy类条件概率
    end
    %得到最大的概率值对应的列数（对应类=列-1）
    [b,b2] = find(label_for_MLE(i,:)==max(label_for_MLE(i,:)));
   % fprintf('%.1f ;%.1f\n ',b,b2);
    label_pre(i,1) = b2-1;%得到对应test类别
    sum = length(find(label_pre(:,1)==test(:,end)));%统计预测正确的test数量
    success_rate = sum/test_num;%正确率
end






