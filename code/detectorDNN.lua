require 'torch';
require 'nn';
require 'image';
cuda = false
require 'cutorch';
require 'cunn';
require 'sys';
require 'csvigo';
local dir = require 'pl.dir';

local c = os.clock()
local t = os.time()

local folder = '/home/andrew/mitosis/data/MITOS/testing/'
--local netPath1 = '/home/andrew/mitosis/data/nets/net.t7'
local netPath1 = '/home/andrew/mitosis/data/nets/dnn1_fullset_aug_20i_lr05_lrd0005_m09_mini200_aeptgl.t7'
local netPath2 = '/home/andrew/mitosis/data/nets/dnn2_fullset_aug_20i_lr05_lrd0005_m09_mini200_aeptgl.t7'

local threshold = 0.1

dofile("getImagePaths.lua")
local imagePaths = getImagePaths(folder)

if paths.dirp(folder .. 'results') == false then
	paths.mkdir(folder .. 'results');
end

dofile("scan.lua")
dofile("expand.lua")

-- define circular kernel
local d = 10
local kernel = torch.FloatTensor(2*d+1,2*d+1):fill(0.001)
for i=1,2*d+1 do
	for j=1,2*d+1 do
		if (i-d-1)^2 + (j-d-1)^2 >= d^2 then
			kernel[i][j] = 0
		end
	end
end

for k,imagePath in ipairs(imagePaths) do
	print(k)

	local c1 = os.clock()
	local t1 = os.time()

	--local windowWidth = 101
	--local windowHeight = 101

	local net1 = torch.load(netPath1)
	local net2 = torch.load(netPath2)
	net1 = expand(net1)
	net2 = expand(net2)
	if cuda then
		net1 = net1:cuda()
		net2 = net2:cuda()
	else
		net1 = net1:float()
		net2 = net2:float()
	end

	local img = image.load(imagePath, 3, 'float')
	if cuda then
		img = img:cuda()
	else
		img = img:float()
	end

	-- scan sixteen versions of the image
	-- four rotations, paired with their reflections, and two neural nets
	--]]
	--[
	local maps = {}
	for i=1,4 do
		for j=1,2 do
			local tmp = image.rotate(img, (i-1)*math.pi/2)
			if j == 2 then
				tmp = image.hflip(tmp)
				tmp = scan(tmp, net1)
				tmp = image.hflip(tmp)
			else
				tmp = scan(tmp, net1)
			end
			maps[(i-1)*2+j] = image.rotate(tmp, -(i-1)*math.pi/2)
		end
	end
	net1 = nil
--[
	for i=5,8 do	
		for j=1,2 do
			local tmp = image.rotate(img, (i-1)*math.pi/2)
			if j == 2 then
				tmp = image.hflip(tmp)
				tmp = scan(tmp, net2)
				tmp = image.hflip(tmp)
			else
				tmp = scan(tmp, net2)
			end
			maps[(i-1)*2+j] = image.rotate(tmp, -(i-1)*math.pi/2)
		end
	end
	net2 = nil
--]]

	-- take the mean of the sixteen maps
	--[
	local sum = maps[1]
	for i=2,#maps do
		sum = sum + maps[i]
	end
	local map = sum/#maps
	
	map = image.convolve(map, kernel, 'same')
	map = map/torch.max(map)					-- normalize

	image.save('test.png',map)
	--map = image.load('test.png',1,'float')

	local results = {}
	local m = 1
	while true do
		local ind = map:eq(map:max()):nonzero()
		local row = ind[1][1]
		local col = ind[1][2]
		local val = map[row][col]
		results[m] = {row, col, val}
		m = m + 1
		for i=-2*d,2*d do
			for j=-2*d,2*d do
				if i^2 + j^2 < 4*d^2 and row+i>0 and col+j>0 and row+i<=map:size(1) and col+j<=map:size(2) then
					map[row+i][col+j] = 0
				end
			end
		end
		if val < threshold then
			break
		end
	end

	local outfile = paths.concat(folder, 'results', paths.basename(imagePath, paths.extname(imagePath)) .. '.csv')
	csvigo.save(outfile, results)

	print(os.clock()-c1)
	print(os.time()-t1)
end

print(os.clock()-c)
print(os.time()-t)
