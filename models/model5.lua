function model2(weights)
	-- define network
	net = nn.Sequential()
	net:add(nn.SpatialConvolution(3, 16, 4, 4))	-- 16x98x98
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))		-- 16x49x49
--	net:add(nn.Dropout())
	net:add(nn.SpatialConvolution(16, 16, 4, 4))	-- 16x46x46
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))		-- 16x23x23
--	net:add(nn.Dropout())
	net:add(nn.SpatialConvolution(16, 16, 4, 4))	-- 16x20x20
	net:add(nn.ReLU())
	net:add(nn.SpatialMaxPooling(2,2,2,2))		-- 16x10x10
--	net:add(nn.Dropout())
	net:add(nn.SpatialConvolution(16, 16, 3, 3))	-- 16x8x8
	net:add(nn.ReLU())
	net:add(nn.SpatialAveragePooling(2,2,2,2))	-- 16x4x4
	net:add(nn.View(16*4*4))
--	net:add(nn.Dropout())
	net:add(nn.Linear(16*4*4, 100))
--	net:add(nn.Dropout())
	net:add(nn.Linear(100, 2))
	net:add(nn.LogSoftMax())
	net = net:cuda()

	-- define loss function
	criterion = nn.ClassNLLCriterion(weights)
	criterion = criterion:cuda()

	return net, criterion
end
