clc,clear;
global imgRow;
global imgCol;
global gamma;
nPerson = 40;%需要读入的人数, 每个人的前五幅图为训练样本，后五幅为测试样本 
k = 80;% 降维至 k 维

img = imread('orl_faces/s1/1.pgm');
[imgRow, imgCol] = size(img);
imshow(img);
[faceContainer, faceLabel] = read_faces(nPerson, 0);%0-读入训练数据
n = size(faceContainer, 1);

%低维空间的图像是(nPerson * 5) * k的矩阵，每行代表一个主成分脸，每个脸20维特征
%读取的数据是200*10304的向量，把每一个像素点当做一维特征，故每张图片有10304维，现对其进行降维
meanVec = mean(faceContainer);
[pcaFace, V] = fastPCA(faceContainer, k, meanVec);

 visualize(V);
 
 %训练特征数据标准化
lowvec = min(pcaFace); 
upvec = max(pcaFace); 
scaledface = scaling(pcaFace, lowvec, upvec);

%SVM样本训练
gamma = 0.005;
c = 100; 

multiSVMstruct = multiSVMTrain(scaledface, nPerson, c);

%测试数据特征降维、标准化
[testFace, testLabel] = read_faces(nPerson, 1); 
m = size(testFace,1); 
for i = 1:m
    testFace(i,:) = testFace(i,:) - meanVec; 
end
pcaTestFace = testFace * V; 
scaledTestFace = scaling(pcaTestFace, lowvec, upvec); 

%SVM样本分类
label = multiSVMTest(scaledTestFace, multiSVMstruct, nPerson); 
accuracy = sum(label == testLabel)/length(label)