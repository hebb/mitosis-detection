function bmp2png(folder)

listing = dir([folder '*.bmp']);

for i=1:size(listing)
    infile = [folder listing(i).name];
    [pathstr, name, ext] = fileparts(infile);
    outfile = [pathstr '/' name '.png'];
    A = imread(infile);
    imwrite(A,outfile)
end