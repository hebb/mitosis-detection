function [len_M, num_true] = add_dataset(image_file, csv_file, dataset_folder)

% parameters
d = 10;
d2 = d*d;
N = 101; % must be odd

R = (N-1)/2;
S = R + d;

% create list of coordinates within a circle of radius d
k = 1;
for i=(1-d):(d-1)
    for j=(1-d):(d-1)
       if i*i + j*j < d2
           region(k,:) = [i j];
           k = k + 1;
       end
    end
end 

% load image and csv files
A = imread(image_file);
M = csvread(csv_file);

[A_height, A_width, ~] = size(A);

% extend the image by mirroring (just for the positive samples)
D = padarray(A,[S S],'symmetric');

% find the centroids
len_M = size(M,1);
P = zeros(len_M,2);
for i=1:len_M;
    tmp = M(i,:);
    tmp = tmp(tmp~=0);
    len_tmp = size(tmp,2);
    sum_X = 0;
    sum_Y = 0;
    for j=1:len_tmp/2
        sum_X = sum_X + tmp(2*j-1);
        sum_Y = sum_Y + tmp(2*j);
    end
    mean_X = 2*sum_X/len_tmp;
    mean_Y = 2*sum_Y/len_tmp;
    P(i,:) = [mean_Y mean_X];
end
M = round(P);

% create positive dataset
% extract many images within a region around each mitosis
len_region = size(region,1);
B = cell(1,len_M*len_region);
for i=1:len_M
    for j=1:len_region
        y1 = M(i,1) + region(j,1) - R + S;
        y2 = M(i,1) + region(j,1) + R + S;
        x1 = M(i,2) + region(j,2) - R + S;
        x2 = M(i,2) + region(j,2) + R + S;
        %{
        if y1 >= 1 && y2 <= A_height && x1 >= 1 && x2 <= A_width
            B{j+(i-1)*len_region} = A(y1:y2, x1:x2, :);
        end
        %}
        B{j+(i-1)*len_region} = D(y1:y2, x1:x2, :);
    end
end
%B = B(~cellfun('isempty', B));

% create negative dataset
H = A_height - N + 1;
W = A_width - N + 1;
num_true = length(B);
C = cell(1,num_true);
for i=1:num_true
    flag = true;
    while flag
        X = randi(W);
        Y = randi(H);
        flag = isPos(X,Y,M,d);
    end
    y1 = Y;
    y2 = Y + 2*R;
    x1 = X;
    x2 = X + 2*R;
    C{i} = A(y1:y2, x1:x2, :);
end

% write set of images to file
n = length(dir([dataset_folder 'true/*.png']));
for i=1:num_true
    file1 = [dataset_folder 'true/' num2str(i+n,'%02u') '.png'];
    file2 = [dataset_folder 'false/' num2str(i+n,'%02u') '.png'];
    imwrite(B{i}, file1, 'png');
    imwrite(C{i}, file2, 'png');
end
