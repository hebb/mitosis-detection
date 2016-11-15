function stainSeparation(folder)

subFolder = 'normalizeStaining';
mkdir([folder subFolder])

listing = dir([folder '*.bmp']);

for i=1:size(listing)
    infile = [folder listing(i).name];
    [pathstr, name, ext] = fileparts(infile);
    normFile = [pathstr '/' subFolder '/' name '_norm.bmp'];
    Hfile = [pathstr '/' subFolder '/' name '_H.bmp'];
    Efile = [pathstr '/' subFolder '/' name '_E.bmp'];
    I = imread(infile);
    [Inorm, H, E] = normalizeStaining(I);
    imwrite(Inorm,normFile)
    imwrite(H,Hfile)
    imwrite(E,Efile)
end