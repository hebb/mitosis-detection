function add_trainset(image_file, csv_file, trainset_folder)

% parameters
d = 10;
d2 = d*d;
N = 101; % must be odd

R = (N-1)/2;

% create list of coordinates within a cirlce of radius d
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

% create positive training set
% extract many images within a region around each mitosis
len_M = size(M,1);
len_region = size(region,1);
B = cell(1,len_M*len_region);
for i=1:len_M
    for j=1:len_region
        y1 = M(i,1) + region(j,1) - R;
        y2 = M(i,1) + region(j,1) + R;
        x1 = M(i,2) + region(j,2) - R;
        x2 = M(i,2) + region(j,2) + R;
        if y1 >= 1 && y2 <= A_height && x1 >= 1 && x2 <= A_width
            B{j+(i-1)*len_region} = A(y1:y2, x1:x2, :);
        end
    end
end
B = B(~cellfun('isempty', B));

% create negative training set
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
    y2 = Y + R;
    x1 = X;
    x2 = X + R;
    C{i} = A(y1:y2, x1:x2, :);
end

% write set of images to file
n = length(dir([trainset_folder 'true/*.tif']));
for i=1:num_true
    file1 = [trainset_folder 'true/' num2str(i+n,'%02u') '.tif'];
    file2 = [trainset_folder 'false/' num2str(i+n,'%02u') '.tif'];
    imwrite(B{i}, file1, 'tif');
    imwrite(C{i}, file2, 'tif');
end
