require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 4, 4))     -- 16x98x98
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2,2,2,2))         -- 16x48x48
net:add(nn.SpatialConvolution(6, 6, 4, 4))     -- 16x46x46
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2,2,2,2))         -- 16x23x23
net:add(nn.SpatialConvolution(6, 6, 4, 4))     -- 16x20x20
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2,2,2,2))         -- 16x10x10
net:add(nn.SpatialConvolution(6, 6, 4, 4))     -- 16x7x7
net:add(nn.Tanh())
net:add(nn.SpatialMaxPooling(2,2,2,2))         -- 16x3x3
net:add(nn.View(16*2*2))
net:add(nn.Linear(16*2*2, 100))
net:add(nn.Tanh())
net:add(nn.Linear(100, 2))
net:add(nn.LogSoftMax())
net = net:cuda()

-- define loss function
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
