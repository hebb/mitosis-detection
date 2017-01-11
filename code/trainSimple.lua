require 'torch';
require 'nn';
require 'optim';

cudaFlag = true

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

--torch.setnumthreads(4)

-- parameters
local batchSize = 100
local learningRate = 0.01
local maxIteration = 10

local folder  = '/home/andrew/mitosis/mitosis-train-old/'

dofile("data.lua")
local classes, classList, imagePaths = getImagePaths(folder)

-- split dataset into training and test sets
local trainClassList = {}
local testClassList = {}
--[
trainClassList[1] = classList[1][{{1,math.ceil(classList[1]:size(1)/2)}}]
trainClassList[2] = classList[2][{{1,math.ceil(classList[2]:size(1)/2)}}]
testClassList[1] = classList[1][{{math.ceil(classList[1]:size(1)/2)+1,classList[1]:size(1)}}]
testClassList[2] = classList[2][{{math.ceil(classList[2]:size(1)/2)+1,classList[2]:size(1)}}]
--]]
--[[
trainClassList[1] = classList[1][{{1,math.ceil(classList[1]:size(1)/20)}}]
trainClassList[2] = classList[2][{{1,math.ceil(classList[2]:size(1)/20)}}]
testClassList[1] = classList[1][{{math.ceil(classList[1]:size(1)/20)+1,math.ceil(classList[1]:size(1)/10)}}]
testClassList[2] = classList[2][{{math.ceil(classList[2]:size(1)/20)+1,math.ceil(classList[1]:size(1)/10)}}]
--]]
--trainClassList = classList

local classRatio = trainClassList[2]:size(1)/trainClassList[1]:size(1)

-- define the model
dofile("/home/andrew/mitosis/models/model.lua")

-- train the network
dofile("train.lua")
train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration, classRatio, false)

-- save the model
torch.save('/home/andrew/mitosis/nets/testNet.t7', net)
--net = torch.load('/home/andrew/mitosis/nets/testNet.t7')

-- test the network
dofile("test.lua")
test(net, classes, testClassList, imagePaths, batchSize)
