%读取ORL人脸库照片里的数据到矩阵（向量化人脸容器）
function [faceContainer, faceLabel] = read_faces(nPerson, flag)
%输入:  nPersons --- 需要读入的人数, 每个人的前五幅图为训练样本，后五幅为测试样本 
%       flag     --- 为0表示读入训练样本，为1表示读入测试样本 
%输出： faceContainer --- 向量化人脸容器，nPerson * 10304 的 2 维矩阵，每行对应一个人脸向量
%       faceLabel是人脸的标签    

    %照片大小
    global imgRow;
    global imgCol;
    %初始化容器和标签
    faceContainer = zeros(nPerson * 5, imgRow * imgCol);
    faceLabel = zeros(nPerson * 5, 1);
  
    for i = 1:nPerson
        %函数num2str(i)说明：将数字转化为字符
        tmpPath = strcat('orl_faces\s', num2str(i), '\');
        for j = 1:5
            path = tmpPath;
            if flag == 0        %训练样本数据
                path = strcat(path, '0' + j);
            else                %测试样本数据
                path = strcat(path, num2str(5 + j));
            end
            path = strcat(path, '.pgm');
            img = imread(path);
            %把读入的图像按列存储为行向量放入向量化人脸容器faceContainer的对应行中
            faceContainer((i - 1) * 5 + j, :) = img(:)';
            faceLabel((i - 1) * 5 + j) = i;
        end
    end
    save('orl_faces\faceContainer','faceContainer');%保存读取的数据
end