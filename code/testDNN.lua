require 'torch';
require 'nn';

cudaFlag = true

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

local folder = '/home/andrew/mitosis/data/mitosis-test/'
local batchSize = 200

dofile("data.lua")

local classes, testClassList, imagePaths = getImagePaths(folder)

--local net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_fcdropout.t7')
--local net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_weightdecay0001.t7')
local net = torch.load('/home/andrew/mitosis/data/nets/dnn2_fullset_aug_20i_lr05_lrd0005_m09_mini200_aeptgl.t7')
--local net = torch.load('/home/andrew/mitosis/data/nets/net.t7')
--local net = torch.load('/home/andrew/mitosis/data/nets/testNet.t7')

-- test the network
dofile("test.lua")
test(net, classes, testClassList, imagePaths, batchSize)
