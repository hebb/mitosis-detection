require 'torch';
require 'nn';

cudaFlag = true

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

local folder = '/home/andrew/mitosis/mitosis-test/'
local batchSize = 200

dofile("data.lua")

local classes, testClassList, imagePaths = getImagePaths(folder)

--local net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_fcdropout.t7')
--local net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_weightdecay0001.t7')
local net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200.t7')
--local net = torch.load('testNet.t7')

-- test the network
dofile("test.lua")
test(net, classes, testClassList, imagePaths, batchSize)
