clc,clear;
train_filename='bird_small.tiff';
data=double(imread('bird_large.tiff'));
data=data/255;
%�������ĸ�����Ҳ���Ǵ���ͼƬ����ɫ����
K = 16;

[x,y,z]=size(data);
X=reshape(data,x*y,z);

%ʹ��ѵ��ͼƬ���о���
%�õ�������16����ɫ
centroids=Ktrain(train_filename);

%�õ�ͼƬ��ÿ�����ص���Ҫʹ�õ���ɫ�±�idx
idx=find_centroids_index(X,centroids,K);
%����ɫ����д��ÿ������
res_data = centroids(idx, :);

%�ָ�ͼƬ��ʽ
large_image = reshape(res_data, x, y, z) .* 255;

%Show
imshow(uint8(round(large_image)))
%Save
imwrite(uint8(round(large_image)), 'bird_kmeans.tiff');