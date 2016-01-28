require 'torch';
require 'nn';
require 'cutorch';

-- load training data
trainset = torch.load('/home/andrew/cifar10torchsmall/cifar10-train.t7')
-- load testing data
testset = torch.load('/home/andrew/cifar10torchsmall/cifar10-test.t7')

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

setmetatable(trainset,
	{__index = function(t, i) 
					return {t.data[i], t.label[i]} 
				end}
);
trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

-- data normalization
mean = {}
stdv  = {}
for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
end

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5)) 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84)) 
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

-- define loss function
criterion = nn.ClassNLLCriterion()

-- train network
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.
trainer:train(trainset)

-- data normalization
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

predicted = net:forward(torch.rand(1,3,32,32))
predicted:exp()
print(predicted)
confidences, indices = torch.sort(predicted, true)
for i=1,indices:size(1) do
	print(confidences[i], indices[i])
end
print(indices[1])

-- test network
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
