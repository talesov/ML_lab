%计算更新样本中心
function centroids = compute_centroids(data, idx, K)
    y = size(data, 2);
    centroids = zeros(K, y);
    for i = 1:K
        tmp = data(idx == i,:);
        centroids(i,:) = sum(tmp)/(size(tmp,1));
    end
end