%��ʾ���ɷַ��������任�ռ��еĻ�����������λ����������
function visualize( B )
%���룺B����ÿ���Ǹ����ɷַ���
%     k�������ɷֵ�ά��
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