    
function centroids=Ktrain(train_filename)
    data = double(imread(train_filename));
    [x,y,z] = size(data);
    data=data/255;%RGB��һ��

    X=reshape(data,x*y,z);
    m = size(X, 1);

    %�������ĸ�����Ҳ���Ǵ���ͼƬ����ɫ����
    K = 16;
    maxiter = 100; %���ĵ�������

    % ��ʼ���������ģ����ѡȡ16��������Ϊ��������
    randidx = randperm(m);
    centroids = X(randidx(1:K), :);

    %K-means ѵ������
    %index=zeros(m,1);

    for step = 1:maxiter
        %�õ���С������±�
        index = find_centroids_index(X, centroids, K);
        %�õ�������ɫֵ
        centroids = compute_centroids(X, index, K);
    end

