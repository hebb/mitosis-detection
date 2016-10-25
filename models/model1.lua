require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 16, 2, 2))		-- 16x100x100
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x50x50
net:add(nn.SpatialConvolution(16, 16, 3, 3))	-- 16x48x48
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x24x24
net:add(nn.SpatialConvolution(16, 16, 3, 3))	-- 16x22x22
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x11x11
net:add(nn.SpatialConvolution(16, 16, 2, 2))	-- 16x10x10
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x5x5
net:add(nn.SpatialConvolution(16, 16, 2, 2))	-- 16x4x4
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x2x2
net:add(nn.View(16*2*2))
net:add(nn.Linear(16*2*2, 100))
net:add(nn.Linear(100, 2))
net:add(nn.LogSoftMax())
net = net:cuda()

-- define loss function
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
