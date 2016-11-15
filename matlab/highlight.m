function out = highlight(image_file, csv_file)

% load image and csv files
A = imread(image_file);
M = csvread(csv_file);
for i=1:3
    B(:,:,i) = rgb2gray(A);
end
[p, q] = size(M);
for i=1:p
    for j=1:q/2
        if M(i,2*j) ~= 0
            col = M(i,2*j-1);
            row = M(i,2*j);
            B(row,col) = A(row,col);
        end
    end
end

out = B;

end