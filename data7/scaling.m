%���ݹ�һ��
function [ scaledFace ] = scaling(faceMatrix, lowvec, upvec)
%�������ݹ淶��
%���� faceMatrix - ��Ҫ���й淶����ͼ������
%     lowvec    -- ԭ������Сֵ
%     upvec     -- ԭ�������ֵ
    [m, n] = size(faceMatrix); 
    scaledFace = zeros(m, n);     
    
    upnew = 1; %����������ÿ�����������ֵ
    lownew = -1; %����������ÿ����������Сֵ

    for i = 1:m 
        scaledFace(i, :) = lownew + (faceMatrix(i, :) - lowvec) ./ (upvec - lowvec) * (upnew - lownew); 
    end 
end
