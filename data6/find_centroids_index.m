%�������Сֵ���±�index
function index=find_centroids_index(data,centroids,K)
    x=size(data,1);
    index=zeros(x,1);
     distance=zeros(K,1);
    
    for i=1:x
        for j=1:K
            %����������ÿ���������ĵľ���
            distance(j)= (data(i, :) - centroids(j, :)) * (data(i, :) - centroids(j, :))';
        end
        %�������Сֵ���±�
        op = find(distance == min(distance));
        index(i) = op(1);
    end
    
end