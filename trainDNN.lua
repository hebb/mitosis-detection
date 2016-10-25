require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

cudaFlag = 1

-- parameters
local batchSize = 1000		-- batch size is approximate; actual batchSizes will vary by +/- N-1, where N is the number of classes
local learningRate = 0.001
local maxIteration = 10 

local folder = '/home/andrew/mitosis/mitosis-train-large2/'

dofile("data.lua")
dofile("train.lua")
dofile("test.lua")

local classes, classList, imagePaths = getImagePaths(folder)

-- split dataset into training and test sets
local trainClassList = {}
local testClassList = {}

-- train with the entire dataset and don't test
--trainClassList = classList

-- train with a subset of the full dataset
--[
trainClassList[1] = classList[1][{{1,math.ceil(classList[1]:size(1)/2)}}]
trainClassList[2] = classList[2][{{1,math.ceil(classList[2]:size(1)/2)}}]
testClassList[1] = classList[1][{{math.ceil(classList[1]:size(1)/2)+1,math.ceil(classList[1]:size(1)/1)}}]
testClassList[2] = classList[2][{{math.ceil(classList[2]:size(1)/2)+1,math.ceil(classList[2]:size(1)/1)}}]
--]]

--[
-- first network
-- define the model
dofile("/home/andrew/mitosis/models/model1.lua")

-- train the network
--net = torch.load('/home/andrew/mitosis/nets/dnn1_halfset_aug_20i_lr001.t7')
--train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)
--torch.save('/home/andrew/mitosis/nets/dnn4_halfset_aug_20i_lr001.t7', net)

-- test the network
--test(classes, testClassList, imagePaths, batchSize, maxIteration)
--]]

-- second network
-- define the model
dofile("/home/andrew/mitosis/models/model2.lua")

-- train the network
net = torch.load('/home/andrew/mitosis/nets/dnn2_halfset_aug_20i_lr001.t7')
dofile("train.lua")
train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)
torch.save('/home/andrew/mitosis/nets/dnn2_halfset_aug_30i_lr001.t7', net)

-- test the network
test(classes, testClassList, imagePaths, batchSize)
