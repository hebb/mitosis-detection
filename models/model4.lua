require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

-- define network
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 16, 6, 6))		-- 16x96x96
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x48x48
net:add(nn.SpatialConvolution(16, 16, 7, 7))	 	-- 16x42x42
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x21x21
net:add(nn.SpatialConvolution(16, 16, 6, 6))		-- 16x16x16
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))			-- 16x8x8
net:add(nn.SpatialConvolution(16, 16, 5, 5))		-- 16x4x4
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
