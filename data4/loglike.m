%����1
%ʵ��������convData.m�ǽ���СΪ12960��nursery.data.txt���ݼ�
%�ָ��һ�ݴ�СΪ10000��ѵ����training_data.txt
%�ʹ�СΪ2960�Ĳ��Լ�test_data.txt
clc,clear;
train = load('training_data.txt');
test = load('test_data.txt');

[train_row,train_col] = size(train);
test_num = size(test,1);
%��¼ÿ�����Ӧ�Ķ�����Ȼ����ֵ�����ֵ��Ӧ���±��1����������Ԥ�����
label_for_MLE = zeros(test_num,5);
label_pre = zeros(test_num,1); %�洢��������Ԥ�����
label_num = max(train(:,end)); %�������
for i = 1:test_num%��1��2960����ͬ������������ÿһ��
    for y = 0:label_num%��0��4����ͬ���
        count_y = length(find(train(:,train_col)==y));%ÿһ�����ĸ���
        p_y = count_y/train_row;%ÿһ�����/����
        log_count_xy = 0;
        for j = 1:train_col-1%��1��8����ͬfeature����sampleÿһ��
            %���Ϊy�����£����ж�Ӧtest.featurej��train.featurej�ĸ���
            count_xy = length(find(train(:,train_col)==y & train(:,j)==test(i,j)));
            %������˹ƽ�������ݲ�ͬ��feature�����������ʣ�yΪ���
            if j==1
                p_xy = (count_xy+1)/(count_y+3);%����+1����ĸ+feature�����
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
            log_count_xy = log_count_xy + log(p_xy);%�˷�ת��Ϊ�ӷ���
            %�������ʸ��ݶ���������ת��Ϊ����
        end
        %��ÿһ��test�����д洢ÿһ�����ĸ���ֵ����Ҷ˹���ӣ���Ϊ��ĸ��ֻͬ���Ƿ���
        label_for_MLE(i,y+1) = log(p_y) + log_count_xy;%py��������ʣ�xy����������
    end
    %�õ����ĸ���ֵ��Ӧ����������Ӧ��=��-1��
    [b,b2] = find(label_for_MLE(i,:)==max(label_for_MLE(i,:)));
   % fprintf('%.1f ;%.1f\n ',b,b2);
    label_pre(i,1) = b2-1;%�õ���Ӧtest���
    sum = length(find(label_pre(:,1)==test(:,end)));%ͳ��Ԥ����ȷ��test����
    success_rate = sum/test_num;%��ȷ��
end






