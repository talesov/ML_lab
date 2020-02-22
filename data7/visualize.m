%显示主成分分量脸（变换空间中的基向量，即单位特征向量）
function visualize( B )
%输入：B——每列是个主成分分量
%     k——主成分的维数
    global imgRow; 
    global imgCol; 
    figure 
    img = zeros(imgRow, imgCol); 
    for i = 1:20 
        img(:) = B(:, i); 
        subplot(4, 5, i); 
        imshow(img, []) 
    end 
end