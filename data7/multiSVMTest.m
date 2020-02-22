function [ label ] = multiSVMTest(testFace, multiSVMstruct, n)
%�Բ������ݽ��з���
    m = size(testFace, 1); 
    voting = zeros(m, n); 
    for i = 1:n-1 %39��label,��i label
        for j = i+1:n  
            label = predict(multiSVMstruct{i}{j}, testFace); 
            voting(:, i) = voting(:, i) + (label == 1); 
            voting(:, j) = voting(:, j) + (label == 0); 
        end
    end
    [~, label] = max(voting,[],2);
end
