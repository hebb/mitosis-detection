function rgb2he(folder)
subFolder = 'H&E';
mkdir([folder subFolder])

bmpListing = dir([folder '*.bmp']);
csvListing = dir([folder '*.csv']);

for i=1:size(bmpListing)
    inFile = [folder bmpListing(i).name];
    [pathstr, name, ext] = fileparts(inFile);
    outFile = [pathstr '/' subFolder '/' name '.bmp'];

    I = imread(inFile);
    [Inorm, H, E] = normalizeStaining(I);
    
    [r, c, p] = size(I);
    HE = zeros(r,c,p,'uint8');
    HE(:,:,1) = rgb2gray(H);
    HE(:,:,2) = rgb2gray(E);
    
    imwrite(HE,outFile)

    copyfile([folder csvListing(i).name], [pathstr '/' subFolder '/' name '.csv']);
end

end

