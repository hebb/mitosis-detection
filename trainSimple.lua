require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

--torch.setnumthreads(4)

-- parameters
local batchSize = 1000
local learningRate = 0.01
local maxIteration = 10

local folder  = '/home/andrew/mitosis/mitosis-train/'

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
--trainClassList = classList

-- define the model
dofile("/home/andrew/mitosis/models/model.lua")

-- train the network
dofile("train.lua")
train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)

-- save the model
torch.save('/home/andrew/mitosis/nets/net.t7', net)
--net = torch.load('/home/andrew/mitosis-detection/model.t7')

-- test the network
dofile("test.lua")
test(classes, testClassList, imagePaths, batchSize)
