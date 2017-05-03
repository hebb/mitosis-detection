require 'torch';
require 'nn';
require 'optim';

--cudaFlag = true
cudaFlag = true

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

-- parameters
local batchSize = 200		-- batch size is approximate; actual batchSizes will vary by +/- N-1, where N is the number of classes
local learningRate = 0.05
local learningRateDecay = 0.0005
local weightDecay = 0.000
local momentum = 0.9
local maxIteration = 20
local augment = true

local trainFolder = '/home/andrew/mitosis/data/mitosis-train-large/'
local testFolder = '/home/andrew/mitosis/data/mitosis-test/'

dofile("data.lua")

local classes, trainClassList, trainImagePaths = getImagePaths(trainFolder)
local classes, testClassList, testImagePaths = getImagePaths(testFolder)


local classRatio = trainClassList[2]:size(1)/trainClassList[1]:size(1)
local weights = torch.Tensor(2)
weights[1] = classRatio
weights[2] = 1

--[
-- first network
-- define the model
dofile("/home/andrew/mitosis/models/model1.lua")
local net, criterion = model1(weights)
if cudaFlag then
	net = net:cuda()
	criterion = criterion:cuda()
end

-- train the network
dofile("train.lua")
--net = torch.load('/home/andrew/mitosis/data/nets/dnn1_halfset_aug_20i_lr001.t7')
--train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)
--torch.save('/home/andrew/mitosis/data/nets/dnn1_fullset_aug_30i_lr05_mini200.t7', net)

-- test the network
dofile("test.lua")
--test(classes, testClassList, imagePaths, batchSize, maxIteration)
--]]


-- second network
-- define the model
dofile("/home/andrew/mitosis/models/model2.lua")
local net, criterion = model2(weights)
if cudaFlag then
	net = net:cuda()
	criterion = criterion:cuda()
end

-- train the network
dofile("train.lua")
net = torch.load('/home/andrew/mitosis/data/nets/model2-pretrained-greedylayerwise2.t7')
train(net, criterion, classes, trainClassList, trainImagePaths, batchSize, learningRate, learningRateDecay, weightDecay, momentum, maxIteration, classRatio, augment, netFolder)
torch.save('/home/andrew/mitosis/data/nets/dnn2_fullset_aug_20i_lr05_lrd0005_m09_mini200_aeptgl.t7', net)

-- test the network
dofile("test.lua")
--net = torch.load('/home/andrew/mitosis/data/nets/dnn2_fullset_aug_20i_lr05_mini200_aept.t7')
test(net, classes, testClassList, testImagePaths, batchSize)
