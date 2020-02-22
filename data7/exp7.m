clc,clear;
global imgRow;
global imgCol;
global gamma;
nPerson = 40;%��Ҫ���������, ÿ���˵�ǰ���ͼΪѵ�������������Ϊ�������� 
k = 80;% ��ά�� k ά

img = imread('orl_faces/s1/1.pgm');
[imgRow, imgCol] = size(img);
imshow(img);
[faceContainer, faceLabel] = read_faces(nPerson, 0);%0-����ѵ������
n = size(faceContainer, 1);

%��ά�ռ��ͼ����(nPerson * 5) * k�ľ���ÿ�д���һ�����ɷ�����ÿ����20ά����
%��ȡ��������200*10304����������ÿһ�����ص㵱��һά��������ÿ��ͼƬ��10304ά���ֶ�����н�ά
meanVec = mean(faceContainer);
[pcaFace, V] = fastPCA(faceContainer, k, meanVec);

 visualize(V);
 
 %ѵ���������ݱ�׼��
lowvec = min(pcaFace); 
upvec = max(pcaFace); 
scaledface = scaling(pcaFace, lowvec, upvec);

%SVM����ѵ��
gamma = 0.005;
c = 100; 

multiSVMstruct = multiSVMTrain(scaledface, nPerson, c);

%��������������ά����׼��
[testFace, testLabel] = read_faces(nPerson, 1); 
m = size(testFace,1); 
for i = 1:m
    testFace(i,:) = testFace(i,:) - meanVec; 
end
pcaTestFace = testFace * V; 
scaledTestFace = scaling(pcaTestFace, lowvec, upvec); 

%SVM��������
label = multiSVMTest(scaledTestFace, multiSVMstruct, nPerson); 
accuracy = sum(label == testLabel)/length(label)