require 'torch';
require 'nn';
require 'image';
require 'cutorch';
require 'cunn';
require 'sys';
require 'csvigo';
local dir = require 'pl.dir';

t = os.clock()

local folder = '/home/andrew/mitosis/MITOS/testing/'
local netPath1 = '/home/andrew/mitosis/nets/dnn1_halfset_aug_20i_lr001.t7'
local netPath2 = '/home/andrew/mitosis/nets/dnn2_halfset_aug_20i_lr001.t7'

N = 8 -- computes confidence at every N pixels, vertically and horizontally
threshold = 0.1

dofile("getImagePaths.lua")
imagePaths = getImagePaths(folder)

if paths.dirp(folder .. 'results') == false then
	paths.mkdir(folder .. 'results');
end

dofile("scan.lua")

-- define circular kernel
d = 10
kernel = torch.FloatTensor(2*d+1,2*d+1):fill(0.001)
for i=1,2*d+1 do
	for j=1,2*d+1 do
		if (i-d-1)^2 + (j-d-1)^2 >= d^2 then
			kernel[i][j] = 0
		else
		end
	end
end

for k,imagePath in ipairs(imagePaths) do
	imagePath = imagePaths[k]

	t1 = os.clock()

	local windowWidth = 101
	local windowHeight = 101

	local img = image.load(imagePath, 3, 'float')
	local net1 = torch.load(netPath1)
	local net2 = torch.load(netPath2)
	net1 = net1:cuda()
	net2 = net2:cuda()

	-- scan eight versions of the image
	-- four rotations and two neural nets
	--]]
	--[
	maps = {}
	for i=1,4 do
		tmp = image.rotate(img, (i-1)*math.pi/2)
		tmp = scan(tmp, net1, windowWidth, windowHeight)
		maps[i] = image.rotate(tmp, -(i-1)*math.pi/2)
	end
--[
	for i=5,8 do	
		tmp = image.rotate(img, (i-1)*math.pi/2)
		tmp = scan(tmp, net2, windowWidth, windowHeight)
		maps[i] = image.rotate(tmp, -(i-1)*math.pi/2)
	end
--]]

	-- take the mean of the eight maps
	--[
	sum = maps[1]
	for i=2,#maps do
		sum = sum + maps[i]
	end
	map = sum/#maps
	
	--[
	map = image.convolve(map, kernel, 'same')
	map = map/torch.max(map)					-- normalize

	image.save('test.png',map)
	--]]

	--[
	results = {}
	m = 1
	while true do
		ind = map:eq(map:max()):nonzero()
		row = ind[1][1]
		col = ind[1][2]
		val = map[row][col]
		results[m] = {row, col, val}
		m = m + 1
		for i=-2*d,2*d do
			for j=-2*d,2*d do
				if i^2 + j^2 < 4*d^2 then
					map[row+i][col+j] = 0
				end
			end
		end
		if val < threshold then
			break
		end
	end
	--]]

	outfile = paths.concat(folder, 'results', paths.basename(imagePath, paths.extname(imagePath)) .. '.csv')
	csvigo.save(outfile, results)

	t2 = os.clock()
	print(t2-t1)

end

print(os.clock()-t)
