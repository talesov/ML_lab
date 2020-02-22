clc,clear;
train_filename='bird_small.tiff';
data=double(imread('bird_large.tiff'));
data=data/255;
%聚类质心个数，也就是代替图片的颜色个数
K = 16;

[x,y,z]=size(data);
X=reshape(data,x*y,z);

%使用训练图片进行聚类
%得到聚类后的16种颜色
centroids=Ktrain(train_filename);

%得到图片中每个像素点需要使用的颜色下标idx
idx=find_centroids_index(X,centroids,K);
%将颜色重新写入每个像素
res_data = centroids(idx, :);

%恢复图片格式
large_image = reshape(res_data, x, y, z) .* 255;

%Show
imshow(uint8(round(large_image)))
%Save
imwrite(uint8(round(large_image)), 'bird_kmeans.tiff');