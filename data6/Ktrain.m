    
function centroids=Ktrain(train_filename)
    data = double(imread(train_filename));
    [x,y,z] = size(data);
    data=data/255;%RGB归一化

    X=reshape(data,x*y,z);
    m = size(X, 1);

    %聚类质心个数，也就是代替图片的颜色个数
    K = 16;
    maxiter = 100; %最大的迭代次数

    % 初始化样本中心，随机选取16个样本作为聚类中心
    randidx = randperm(m);
    centroids = X(randidx(1:K), :);

    %K-means 训练过程
    %index=zeros(m,1);

    for step = 1:maxiter
        %得到最小距离的下标
        index = find_centroids_index(X, centroids, K);
        %得到聚类颜色值
        centroids = compute_centroids(X, index, K);
    end

