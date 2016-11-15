function train(net, criterion, classes, classList, imagePaths, batchSize, learningRate, maxIteration, classRatio, augment, netFolder)
	c = os.clock()
	t = os.time()

	dofile("randRotateMirror.lua")

	-- compute size of each batch
	batchSizes, numBatches = getBatchSizes(classes,classList, batchSize)	

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

	local iteration = 1

	while true do
		local currentError = 0
--[
		local sampleSum = {}
		for i=1,#classes do
			sampleSum[i] = 0
		end
--]]
--[
		for i=1,numBatches do
			-- split trainClassList into batches
--[
			local sampleList = {}
			for j=1,#classes do
				sampleList[j] = classList[j][{{sampleSum[j] + 1, sampleSum[j] + batchSizes[j][i]}}]
				sampleSum[j] = sampleSum[j] + batchSizes[j][i]
			end
--]]

			-- get dataset from sampleList
			local dataset = getSample(classes, sampleList, imagePaths)

			-- or get a random batch
--			local dataset = getRandomSample(classes, batchSize, classList, imagePaths)
			
			-- augment the training set with random rotations and mirroring	
			if augment then
				dataset = randRotateMirror(dataset)
			end

			dataset.data = dataset.data:cuda()
			dataset.label = dataset.label:cuda()
--[[
			for t = 1, dataset:size() do
				perm = torch.randperm(dataset:size())
				local example = dataset[perm[t]]
--[[				local input = example[1]
				local target = example[2]

				input = input:cuda()

				currentError = currentError + criterion:forward(net:forward(input), target)

				net:updateGradInput(input, criterion:updateGradInput(net.output, target))
--[[
				if target == 1 then
					net:accUpdateGradParameters(input, criterion.gradInput, learningRate*classRatio)
				else
					net:accUpdateGradParameters(input, criterion.gradInput, learningRate)
				end
--]]
--[[				net:accUpdateGradParameters(input, criterion.gradInput, learningRate)
			end
--]]

--[[
			local example = dataset[1]
			local input = example[1]
			local target = example[2]
--]]
			local input = dataset.data
			local target = dataset.label

			currentError = currentError + criterion:forward(net:forward(input), target)

			net:updateGradInput(input, criterion:updateGradInput(net.output, target))
			net:accUpdateGradParameters(input, criterion.gradInput, learningRate)
--]]
		end

		netFile = netFolder .. 'iter' .. iteration .. '.t7'
		torch.save(netFile, net)

		currentError = currentError/numBatches
		print("# current error = " .. currentError)
		iteration = iteration + 1
		if maxIteration > 0 and iteration > maxIteration then
			print("# StochasticGradient: you have reached the maximum number of iterations")
			print("# training error = " .. currentError)
			break
		end
	end

	print(os.clock()-c)
	print(os.time()-t)

end
