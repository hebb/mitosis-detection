require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 16, 6, 6))		-- 16x96x96
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x48x48
net:add(nn.SpatialConvolution(16, 16, 5, 5))	 	-- 16x44x44
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x22x22
net:add(nn.SpatialConvolution(16, 16, 5, 5))		-- 16x18x18
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x9x9
net:add(nn.SpatialConvolution(16, 16, 4, 4))		-- 16x6x6
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x3x3
net:add(nn.View(16*3*3))
net:add(nn.Linear(16*3*3, 100))
net:add(nn.Linear(100, 2))
net:add(nn.LogSoftMax())
net = net:cuda()

-- define loss function
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
