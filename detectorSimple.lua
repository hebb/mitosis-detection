require 'torch';
require 'nn';
require 'image';
require 'cutorch';
require 'cunn';
require 'sys';
local dir = require 'pl.dir';

t = os.clock()

local trainFolder = '/home/andrew/mitosis-detection/MITOS/training/'
local mapFolder = '/home/andrew/mitosis-detection/maps/'

local modelPath = '/home/andrew/mitosis-detection/model.t7'

N = 2 -- computes confidence at every N pixels, vertically and horizontally

dofile("getImagePaths.lua")
imagePaths = getImagePaths(trainFolder)

if paths.dirp(mapFolder) == false then
	paths.mkdir(mapFolder);
end

dofile("scan.lua")

for k,imagePath in ipairs(imagePaths) do

	print(k)
	t1 = os.clock()

	local windowWidth = 101
	local windowHeight = 101

	local img = image.load(imagePath, 3, 'float')
	local net = torch.load(modelPath)
	net = net:cuda()

	map = scan(img, net, windowWidth, windowHeight)

	outfile = paths.concat(mapFolder, paths.basename(imagePath))
	image.save(outfile, map)

	print(os.clock()-t1)

end

print(os.clock()-t)
