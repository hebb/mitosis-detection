require 'torch';
require 'nn';
require 'optim';

--cudaFlag = true
cudaFlag = false

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

-- parameters
local batchSize = 200		-- batch size is approximate; actual batchSizes will vary by +/- N-1, where N is the number of classes
local learningRate = 0.05
local maxIteration = 30
local augment = true

local folder = '/home/andrew/mitosis/mitosis-train-large/'

dofile("data.lua")

local classes, classList, imagePaths = getImagePaths(folder)

-- split dataset into training and test sets
local trainClassList = {}
local testClassList = {}

-- train with the entire dataset and don't test
trainClassList = classList
testClassList = classList

-- train with a subset of the full dataset
--[[
trainClassList[1] = classList[1][{{1,math.ceil(classList[1]:size(1)/2)}}]
trainClassList[2] = classList[2][{{1,math.ceil(classList[2]:size(1)/2)}}]
testClassList[1] = classList[1][{{math.ceil(classList[1]:size(1)/2)+1,math.ceil(classList[1]:size(1)/1)}}]
testClassList[2] = classList[2][{{math.ceil(classList[2]:size(1)/2)+1,math.ceil(classList[2]:size(1)/1)}}]
--]]

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
--net = torch.load('/home/andrew/mitosis/nets/dnn1_halfset_aug_20i_lr001.t7')
--train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)
--torch.save('/home/andrew/mitosis/nets/dnn1_fullset_aug_30i_lr05_mini200.t7', net)

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
--net = torch.load('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_weightdecay1.t7')
train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration, classRatio, augment, netFolder)
torch.save('/home/andrew/mitosis/nets/dnn2_fullset_aug_30i_lr05_mini200_weightdecay0001.t7', net)

-- test the network
dofile("test.lua")
test(net, classes, testClassList, imagePaths, batchSize)
