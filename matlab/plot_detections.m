function plot_detections(imageFile, csvFile, threshold)

A = imread(imageFile);
results = csvread(csvFile);

% cut results off at the threshold
for i=1:size(results,1)
    if results(i,3) < threshold
        results = results(1:i-1,:);
            break
    end
end

% mark detections
for i=1:size(results,1)
    A = insertShape(A, 'FilledCircle', [results(i,2) results(i,1) 10], 'Opacity', 1);
end

imwrite(A,'detections.bmp')