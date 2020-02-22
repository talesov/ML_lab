%将数据转换为28*28像素的图像并将所有784像素点展开成一行
function svm = re_hand_digits(filename,n)
  fidin = fopen(filename); 
  i = 1;
  apres = [];

while ~feof(fidin)
  tline = fgetl(fidin); % 从文件读行 
  apres{i} = tline;
  i = i+1;
end

 grid = zeros(n,784);%28*28像素
 label = zeros(n,1);
 for k = 1:n
    a = char(apres(k));
    lena = size(a,2);%n=2返回列数
    xy = sscanf(a(4:lena), '%d:%d');
    label(k,1) = sscanf(a(1:3),'%d');
    lenxy = size(xy,1);%n=1返回行数
    for i=2:2:lenxy  %% 隔一个数
      if(xy(i)<=0)
          break
      end
      grid(k,xy(i-1)) = xy(i) * 100/255; %转为有颜色的图像
    end  
 end 
  svm.grid = grid;
  svm.label = label;
end

