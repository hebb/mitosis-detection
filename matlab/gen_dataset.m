clear
close all

% path to folder containing training data
old_trainset_folder = '/home/andrew/mitosis/MITOS/training/';

% make training set directories
new_trainset_folder = '/home/andrew/mitosis/mitosis-train/';
if exist(new_trainset_folder, 'dir')
    rmdir(new_trainset_folder, 's')
end
mkdir([new_trainset_folder 'true'])
mkdir([new_trainset_folder 'false'])

% count the total number of images for the waitbar
N = 0;
for j=1:5
   % find the number of images in the folder
    folder = [old_trainset_folder 'A' num2str(j-1, '%02u') '_v2/'];
    n = length(dir([folder '*.csv']));
    N = N + n;
end

% iterate over the twelve patients
M = 0;
P = 0;
k = 0;
h = waitbar(0,'Creating training set ... 0 %');
for j=1:5
    % find the number of images in the folder
    folder = [old_trainset_folder 'A' num2str(j-1, '%02u') '_v2/'];
    image_files = dir([folder '*.bmp']);
    csv_files = dir([folder '*.csv']);
    n = length(image_files);

    % create new data set
    for i=1:n
        image_file = [folder image_files(i).name];
        csv_file = [folder csv_files(i).name];
        [m, p] = add_dataset(image_file, csv_file, new_trainset_folder);
        M = M + m;
        P = P + p;
        k = k + 1;
        waitbar(k/N,h,['Creating training set ... ' num2str(100*k/N) ' %']);
    end
end

close(h)

disp(['Created ' num2str(P) ' window training images from ' num2str(N) ' large training images containing ' num2str(M) ' mitotic figures.'])