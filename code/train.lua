function train(net, criterion, classes, classList, imagePaths, batchSize, learningRate, learningRateDecay, weightDecay, momentum, maxIteration, classRatio, augment)
	c = os.clock()
	t = os.time()

	dofile("randRotateMirror.lua")

	-- compute size of each batch
	batchSizes, numBatches = getBatchSizes(classes, classList, batchSize)	

	-- shuffle the images 
	
	local temp = {}
	for i=1,#classes do
		local perm = torch.randperm(classList[i]:size(1))
		temp[i] = torch.LongTensor(classList[i]:size(1)) 
		for j=1,classList[i]:size(1) do
			temp[i][j] = classList[i][perm[j]]
		end
	end
	classList = temp

	-- train
	print("# StochasticGradient: training")

	net:training()

	params, gradParams = net:getParameters()
	optimState = {}
	optimState.learningRate = learningRate
	optimState.learningRateDecay = learningRateDecay
	optimState.weightDecay = weightDecay
	optimState.momentum = momentum

	--while true do
	for epoch = 1, maxIteration do
		c1 = os.clock()
		t1 = os.time()

		local currentError = 0

		local sampleSum = {}
		for i=1,#classes do
			sampleSum[i] = 0
		end

		for i=1,numBatches do
			t2 = os.time()
			c2 = os.clock()

			-- split classList into batches
			local sampleList = {}
			for j=1,#classes do
				sampleList[j] = classList[j][{{sampleSum[j] + 1, sampleSum[j] + batchSizes[j][i]}}]
				sampleSum[j] = sampleSum[j] + batchSizes[j][i]
			end

			-- get dataset from sampleList
			local dataset = getSample(classes, sampleList, imagePaths)

			-- or get a random batch
--			local dataset = getRandomSample(classes, batchSize, classList, imagePaths)
			
			-- augment the training set with random rotations and mirroring	
			if augment then
				dataset = randRotateMirror(dataset)
			end

			if cudaFlag then
				dataset.data = dataset.data:cuda()
				dataset.label = dataset.label:cuda()
			end

			local input = dataset.data
			local target = dataset.label

			function feval(params)
				gradParams:zero()

				local outputs = net:forward(input)
				local loss = criterion:forward(outputs, target)
				local dloss_doutputs = criterion:backward(outputs, target)
				net:backward(input, dloss_doutputs)

				return loss, gradParams
			end
			_, fs = optim.sgd(feval, params, optimState)

			print('Epoch = ' .. epoch .. ' of ' .. maxIteration)
			print('Batch = ' .. i .. ' of ' .. numBatches)
			print('Error = ' .. fs[1])
			print('CPU batch time = ' .. os.clock()-c2 .. ' seconds')
			print('Actual batch time (rounded) = ' .. os.time()-t2 .. ' seconds')
			if epochClock then
				print('CPU epoch time = ' .. epochClock .. ' seconds')
				print('Actual epoch time (rounded) = ' .. epochTime .. ' seconds')
			end
			print('')
		end
		
		epochClock = os.clock()-c1
		epochTime = os.time()-t1
	end

	totalClock = os.clock()-c
	totalTime = os.time()-t
	print('Total CPU time = ' .. totalClock .. ' seconds')
	print('Total actual time (rounded) ' .. totalTime .. ' seconds')
end
