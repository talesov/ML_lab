%求距离最小值的下标index
function index=find_centroids_index(data,centroids,K)
    x=size(data,1);
    index=zeros(x,1);
     distance=zeros(K,1);
    
    for i=1:x
        for j=1:K
            %计算样本到每个聚类中心的距离
            distance(j)= (data(i, :) - centroids(j, :)) * (data(i, :) - centroids(j, :))';
        end
        %求距离最小值的下标
        op = find(distance == min(distance));
        index(i) = op(1);
    end
    
end