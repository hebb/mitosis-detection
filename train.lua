function train(net, criterion, classes, trainClassList, imagePaths, batchSize, learningRate, maxIteration)
	t = os.clock()

	-- compute size of each batch
	-- alternate rounding direction to prevent error accumulation and bound the size of the last batch
	local numSamples = 0
	for i=1,#classes do
		numSamples = numSamples + trainClassList[i]:nElement()
	end
	local numBatches = math.ceil(numSamples/batchSize)

--[[
	local batchSizes = {}
	for i=1,#classes do
		local roundFlag = 0
		local batchSum = 0
		batchSizes[i] = {}
		for j=1,numBatches-1 do
			if roundFlag == 0 then
				batchSizes[i][j] = math.floor(trainClassList[i]:nElement()/numBatches)
			else
				batchSizes[i][j] = math.ceil(trainClassList[i]:nElement()/numBatches)
			end
			
			batchSum = batchSum + batchSizes[i][j]
			
			if j*trainClassList[i]:nElement()/numBatches > batchSum then
				roundFlag = 1
			else
				roundFlag = 0
			end
		end
		batchSizes[i][numBatches] = trainClassList[i]:nElement() - batchSum
	end
--]]

	-- train
	print("# StochasticGradient: training")

	local iteration = 1

	while true do
		local currentError = 0
		
--[[
		local sampleSum = {}
		for i=1,#classes do
			sampleSum[i] = 0
		end
--]]

--[
		for i=1,numBatches do
			-- split trainClassList into batches
--[[
			local sampleList = {}
			for j=1,#classes do
				sampleList[j] = trainClassList[j][{{sampleSum[j] + 1, sampleSum[j] + batchSizes[j][i]}}]
				sampleSum[j] = sampleSum[j] + batchSizes[j][i]

			end
--]]

			-- get dataset from sampleList
--			local dataset = getSample(classes, sampleList, imagePaths)
			local dataset = getRandomSample(classes, batchSize, trainClassList, imagePaths)
			dataset.data = dataset.data:cuda()
			dataset.label = dataset.label:cuda()

--[
			for t = 1,dataset:size() do
				perm = torch.randperm(dataset:size())
				local example = dataset[perm[t]]
--[
				local input = example[1]
				local target = example[2]

	
-- augment the training set with random rotations and mirroring			
--[
				theta = math.pi/2*torch.random(0,3)
				input = input:double()
				input = image.rotate(input, theta)
				if torch.random(1,2) == 1 then
					input = image.hflip(input)	
				end
				if cudaFlag == 1 then
					input = input:cuda()
				end
--]

				currentError = currentError + criterion:forward(net:forward(input), target)

				net:updateGradInput(input, criterion:updateGradInput(net.output, target))
				net:accUpdateGradParameters(input, criterion.gradInput, learningRate)
			end
--]]
--[[
			local example = dataset[1]
			local input = example[1]
			local target = example[2]
			--local input = dataset[1].data
			--local target = dataset[1].label

			currentError = currentError + criterion:forward(net:forward(input), target)

			net:updateGradInput(input, criterion:updateGradInput(net.output, target))
			net:accUpdateGradParameters(input, criterion.gradInput, learningRate)
--]]

		

		end

		print("# current error = " .. currentError)
		iteration = iteration + 1
		if maxIteration > 0 and iteration > maxIteration then
			print("# StochasticGradient: you have reached the maximum number of iterations")
			print("# training error = " .. currentError)
			break
		end
	end

	print(os.clock()-t)

end
