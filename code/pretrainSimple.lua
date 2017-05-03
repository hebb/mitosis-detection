require 'torch';
require 'nn';
require 'optim';

cudaFlag = true

if cudaFlag then
	require 'cutorch';
	require 'cunn';
end

c = os.clock()
t = os.time()

-- parameters
local learningRate = 0.05
local learningRateDecay = 0.0005
local weightdecay = 0.0000
local momentum = 0.9
local maxIteration = 5
local p = 0.25
local batchSize = 200


local folder = '/home/andrew/mitosis/data/mitosis-train-large'

dofile("data.lua")
local classes, classList, imagePaths = getImagePaths(folder)

dofile('/home/andrew/mitosis/models/model2.lua')
net = model2()
if cudaFlag then
	net = net:cuda()
end
dofile('/home/andrew/mitosis/code/autoencoder.lua')
autoencoder = convnet2autoencoder(net)

--[
autoencoder:insert(nn.Dropout(p),1)
if cudaFlag then
	autoencoder = autoencoder:cuda()
end

criterion = nn.MSECriterion()
if cudaFlag then
	criterion = criterion:cuda()
end

-- compute size of each batch
batchSizes, numBatches = getBatchSizes(classes, classList, batchSize)

-- shuffle the images
classList = shuffleImages(classList, classes)

-- train
print("# StochasticGradient: training")

autoencoder:training()

subNet = nn.Sequential()
subNet:insert(autoencoder:get(1),1)

-- count the number of convolutional layers
numConvLayers = 0
for i = 1, net:size() do
	if torch.typename(net:get(i)) == 'nn.SpatialConvolution' then
		numConvLayers = numConvLayers + 1
	end
end

errors = {}
for j = 1, numConvLayers do
	c0 = os.clock()
	t0 = os.time()
--[
	subNet:insert(autoencoder:get(3*(j-1)+2),3*(j-1)+2)
	subNet:insert(autoencoder:get(3*(j-1)+3),3*(j-1)+3)
	subNet:insert(autoencoder:get(3*(j-1)+4),3*(j-1)+4)
	subNet:insert(autoencoder:get(autoencoder:size()-3*(j-1)-2),3*(j-1)+5)
	subNet:insert(autoencoder:get(autoencoder:size()-3*(j-1)-1),3*(j-1)+6)
	subNet:insert(autoencoder:get(autoencoder:size()-3*(j-1)),3*(j-1)+7)

	params, gradParams = subNet:getParameters()

	optimState = {}
	optimState.learningRate = learningRate
	optimState.learningRateDecay = learningRateDecay
	optimState.weightDecay = weightDecay
	optimState.momentum = momentum
--]]

	for epoch = 1, maxIteration do
		c1 = os.clock()
		t1 = os.time()

				
		local currentError = 0

		local sampleSum = {}
		for i = 1, #classes do
			sampleSum[i] = 0
		end

		for i = 1, numBatches do
			c2 = os.clock()
			t2 = os.time()

			-- split classList into batches
			local sampleList = {}
			for j=1,#classes do
				sampleList[j] = classList[j][{{sampleSum[j] + 1, sampleSum[j] + batchSizes[j][i]}}]
				sampleSum[j] = sampleSum[j] + batchSizes[j][i]
			end

			local dataset = getSample(classes, sampleList, imagePaths)
			if cudaFlag then
				dataset.data = dataset.data:cuda()
			end
			dataset.label = dataset.data

			local input = dataset.data
			local target = dataset.label

			function feval(params)
				gradParams:zero()

				local outputs = subNet:forward(input)
				local loss = criterion:forward(outputs, target)
				local dloss_doutputs = criterion:backward(outputs, target)
				subNet:backward(input, dloss_doutputs)

				return loss, gradParams
			end
			 _, fs = optim.sgd(feval, params, optimState)

			print('Layer = ' .. j .. ' of ' .. numConvLayers)
			print('Epoch = ' .. epoch .. ' of ' .. maxIteration)
			print('Batch = ' .. i .. ' of ' .. numBatches)
			for k=1,#errors do
				print('Final Error for Layer ' .. k .. ' = ' .. errors[k])
			end
			print('Error = ' .. fs[1])
			print('CPU batch time = ' .. os.clock()-c2 .. ' seconds')
			print('Actual batch time (rounded) = ' .. os.time()-t2 .. ' seconds')
			if epochClock then
				print('CPU epoch time = ' .. epochClock .. ' seconds')
				print('Actual epoch time (rounded) = ' .. epochTime .. ' seconds')
			end
			if layerClock then
				print('CPU layer time = ' .. layerClock .. ' seconds')
				print('Actual layer time (rounded) = ' ..layerTime .. ' seconds')
			end
			print('Total CPU time so far = ' .. os.clock()-c .. ' seconds')
			print('Total actual time so far (rounded) = ' .. os.time()-t .. ' seconds')
			print('')
		end

		epochClock = os.clock()-c1
		epochTime = os.time()-t1
	end

	errors[j] = fs[1]

	layerClock = os.clock()-c0
	layerTime = os.time()-t0
end

--autoencoder = torch.load('/home/andrew/mitosis/data/nets/pretrain.t7')

-- get CNN from autoencoder
net = autoencoder2convnet(autoencoder,net)

torch.save('/home/andrew/mitosis/data/nets/model2-pretrained-greedylayerwise2.t7',net)

totalClock = os.clock()-c
totalTime = os.time()-t
print('Total CPU time = ' .. totalClock .. ' seconds')
print('Total actual time (rounded) ' .. totalTime .. ' seconds')
