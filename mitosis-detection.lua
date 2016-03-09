require 'torch';
require 'nn';
require 'cutorch';

--[[
-- load training data
trainset = torch.load('/home/andrew/cifar10torchsmall/cifar10-train.t7')
-- load testing data
testset = torch.load('/home/andrew/cifar10torchsmall/cifar10-test.t7')

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

for i=1,10000 do
	if trainset.label[i] >= 3 and trainset.label[i] <= 8 then
		trainset.label[i] = 1
	else
		trainset.label[i] = 2
	end
	if testset.label[i] >= 3 and testset.label[i] <= 8 then
		testset.label[i] = 1
	else
		testset.label[i] = 2
	end
end
--]]

--frog = trainset.data[1]
--truck = trainset.data[2]
frog = torch.rand(3, 101, 101)
truck = torch.rand(3, 101, 101)

trainset = {}
trainset.data = torch.zeros(10000, 3, 101, 101)
trainset.label = torch.ones(10000)
trainset.label[{{1,5000}}] = trainset.label[{{1,5000}}]*1
trainset.label[{{5001,10000}}] = trainset.label[{{5001,10000}}]*2
trainset.label = trainset.label:byte()

testset = {}
testset.data = torch.zeros(10000, 3, 101, 101)
testset.label = torch.ones(10000)
testset.label[{{1,5000}}] = testset.label[{{1,5000}}]*1
testset.label[{{5001,10000}}] = testset.label[{{5001,10000}}]*2
testset.label = testset.label:byte()

for i=1,5000 do
	trainset.data[i] = frog
	testset.data[i] = frog
	--trainset.data[i] = torch.rand(3, 32, 32)
	--testset.data[i] = torch.rand(3, 32, 32)
end
for i=5001,10000 do
	trainset.data[i] = truck
	testset.data[i] = truck
	--trainset.data[i] = torch.rand(3, 32, 32)
	--testset.data[i] = torch.rand(3, 32, 32)
end

classes = {'animal', 'vehicle'}

setmetatable(trainset,
	{__index = function(t, i) 
					return {t.data[i], t.label[i]} 
				end}
);

function trainset:size() 
    return self.data:size(1) 
end

-- data normalization
trainset.data = trainset.data:double()
mean = {}
stdv  = {}
for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

testset.data = testset.data:double()
for i=1,3 do
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

-- resize the data
-- trainset.data = trainset.data:resize(10000,3,101,101)

-- define network

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 16, 2, 2))		-- 16x100x100
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x50x50
net:add(nn.SpatialConvolution(16, 16, 3, 3))	-- 16x48x48
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x24x24
net:add(nn.SpatialConvolution(16, 16, 3, 3))	-- 16x22x22
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x11x11
net:add(nn.SpatialConvolution(16, 16, 2, 2))	-- 16x10x10
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x5x5
net:add(nn.SpatialConvolution(16, 16, 2, 2))	-- 16x4x4
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x2x2
net:add(nn.View(16*2*2))
net:add(nn.Linear(16*2*2, 100))
net:add(nn.Linear(100, 2))
net:add(nn.LogSoftMax())

--[[
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 2))
net:add(nn.LogSoftMax())
--]]

-- define loss function
criterion = nn.ClassNLLCriterion()

-- train network
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)

-- resize the data
-- testset.data = testset.data:resize(10000,3,101,101)

-- test network
--[[
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
		correct = correct + 1
    end
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
print(100*correct/10000 .. ' %')
--]]

class_performance = {0, 0}
correct = 0
class_number = {0, 0}
for i=1,10000 do
	local groundtruth = testset.label[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
		correct = correct + 1
    end
	class_number[groundtruth] = class_number[groundtruth] + 1
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/class_number[i] .. ' %')
end
print(100*correct/10000 .. ' %')

