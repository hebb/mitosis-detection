function [len_M, num_false] = add_dataset2(image_file, csv_file, dataset_folder, map_file)

% parameters
% num_false = 26685;
num_false = 26602;
d = 10;
d2 = d*d;
N = 101; % must be odd

R = (N-1)/2;

% load image and csv files
A = imread(image_file);
M = csvread(csv_file);
map = imread(map_file);

[A_height, A_width, ~] = size(A);

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

% create negative dataset
H = A_height - N + 1;
W = A_width - N + 1;
B = cell(1,num_false);
for i=1:num_false
    flag = true;
    while flag
        X = randi(W) + R;
        Y = randi(H) + R;
        if randi([0,255]) < map(Y,X)
            flag = isPos(X,Y,M,d);
        end
    end
    y1 = Y - R;
    y2 = Y + R;
    x1 = X - R;
    x2 = X + R;
    B{i} = A(y1:y2, x1:x2, :);
end

% write set of images to file
n = length(dir([dataset_folder '*.png']));
for i=1:num_false
    file = [dataset_folder num2str(i+n,'%02u') '.png'];
    imwrite(B{i}, file, 'png');
end
