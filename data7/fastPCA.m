% 快速PCA
function [pcaA, V] = fastPCA(A, k, meanVec)
% 输入： A --- 样本矩阵，每行为一个样本
%       k --- 降维至 k 维
%       meanVec --- 样本均值
% 输出：pcaA --- 降维后的 k 维样本特征向量组成的矩阵，每行一个样本，列数 k 为降维后的样本特征维数
%      V --- 主成分向量

    %(一)Step one: 中心化数据  
    %样本矩阵中心化，每一维度减去该维度的均值，使得每一维度的均值为0
    %repmat:Replicate Matrix复制和平铺矩阵
    m = size(A, 1);
    Z = (A - repmat(meanVec, m, 1));
    %(二)Step two: 求中心化数据的协方差矩阵
    T = Z * Z';            % 计算协方差矩阵的转置
    %(三)Step three: 计算特征向量和特征值
    % 计算 T 的前 k 个特征值和特征向量
    [V, ~] = eigs(T, k);   %V为m*k, k个特征向量
    V = Z' * V;            % 得到协方差矩阵 T' 的特征向量
    for i = 1:k            % 特征向量归一化为单位特征向量
        %norm 为范数，默认为2范数(各分量的平方和 再开根号)
        V(:, i) = V(:, i) / norm(V(:, i));
    end
    pcaA = Z * V;          % 线性变换（投影）降维至 k 维
end