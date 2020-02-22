%��ȡORL��������Ƭ������ݵ���������������������
function [faceContainer, faceLabel] = read_faces(nPerson, flag)
%����:  nPersons --- ��Ҫ���������, ÿ���˵�ǰ���ͼΪѵ�������������Ϊ�������� 
%       flag     --- Ϊ0��ʾ����ѵ��������Ϊ1��ʾ����������� 
%����� faceContainer --- ����������������nPerson * 10304 �� 2 ά����ÿ�ж�Ӧһ����������
%       faceLabel�������ı�ǩ    

    %��Ƭ��С
    global imgRow;
    global imgCol;
    %��ʼ�������ͱ�ǩ
    faceContainer = zeros(nPerson * 5, imgRow * imgCol);
    faceLabel = zeros(nPerson * 5, 1);
  
    for i = 1:nPerson
        %����num2str(i)˵����������ת��Ϊ�ַ�
        tmpPath = strcat('orl_faces\s', num2str(i), '\');
        for j = 1:5
            path = tmpPath;
            if flag == 0        %ѵ����������
                path = strcat(path, '0' + j);
            else                %������������
                path = strcat(path, num2str(5 + j));
            end
            path = strcat(path, '.pgm');
            img = imread(path);
            %�Ѷ����ͼ���д洢Ϊ������������������������faceContainer�Ķ�Ӧ����
            faceContainer((i - 1) * 5 + j, :) = img(:)';
            faceLabel((i - 1) * 5 + j) = i;
        end
    end
    save('orl_faces\faceContainer','faceContainer');%�����ȡ������
end