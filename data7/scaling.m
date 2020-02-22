%数据归一化
function [ scaledFace ] = scaling(faceMatrix, lowvec, upvec)
%特征数据规范化
%输入 faceMatrix - 需要进行规范化的图像数据
%     lowvec    -- 原来的最小值
%     upvec     -- 原来的最大值
    [m, n] = size(faceMatrix); 
    scaledFace = zeros(m, n);     
    
    upnew = 1; %所有数据中每个特征的最大值
    lownew = -1; %所有数据中每个特征的最小值

    for i = 1:m 
        scaledFace(i, :) = lownew + (faceMatrix(i, :) - lowvec) ./ (upvec - lowvec) * (upnew - lownew); 
    end 
end
