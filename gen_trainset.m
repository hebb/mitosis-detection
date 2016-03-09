clear
close all

% path to folder containing training data
old_trainset_folder = '/home/andrew/mitosis-detection/training/';

% make training set directories
new_trainset_folder = '/home/andrew/mitosis-detection/trainset/';
rmdir(new_trainset_folder, 's')
mkdir([new_trainset_folder 'true'])
mkdir([new_trainset_folder 'false'])

% iterate over the twelve patients
h = waitbar(0,['Creating training set ... 0 %']);
for j=1:12
    % find the number of images in the folder
    folder = [old_trainset_folder num2str(j, '%02u') '/'];
    n = length(dir([folder '*.tif']));

    % create new data set
    for i=1:n
        image_file = [folder num2str(i,'%02u') '.tif'];
        csv_file = [folder num2str(i,'%02u') '.csv'];
        if exist(csv_file, 'file')
            add_trainset(image_file, csv_file, new_trainset_folder);
        end
    end
    waitbar(j/12,h,['Creating training set ... ' num2str(100*j/12) ' %']);
end
close(h)