require 'torch';
require 'nn';
require 'image';
cuda = false
--if cuda then
	require 'cutorch';
	require 'cunn';
--end
require 'sys';
local dir = require 'pl.dir';

local c = os.clock()
local t = os.time()

local trainFolder = '/home/andrew/mitosis/MITOS/training/'
local mapFolder = '/home/andrew/mitosis/maps/'
local netPath = '/home/andrew/mitosis/nets/net.t7'

dofile("getImagePaths.lua")
imagePaths = getImagePaths(trainFolder)

if paths.dirp(mapFolder) == false then
	paths.mkdir(mapFolder)
end

dofile("scan.lua")
dofile("expand.lua")

for k,imagePath in ipairs(imagePaths) do
	print(k)
	
	local c1 = os.clock()
	local t1 = os.time()

	local net = torch.load(netPath)
	net = expand(net)
	if cuda then
		net = net:cuda()
	else
		net = net:float()
	end

	local img = image.load(imagePath, 3, 'float')
	if cuda then
		img = img:cuda()
	else
		img = img:float()
	end
	local map = scan(img, net)
	local outfile = paths.concat(mapFolder, paths.basename(imagePath))
	
	image.save(outfile, map)

	print(os.clock()-c1)
	print(os.time()-t1)

end

print(os.clock()-c)
print(os.time()-t)
