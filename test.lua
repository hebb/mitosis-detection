require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';

function test(classes, testClassList, imagePaths, batchSize)
	class_performance = {0, 0}
	correct = 0
	class_number = {0, 0}
	numSamples = 0

	-- compute size of each batch
	-- alternate rounding direction to prevent error accumulation and bound the size of the last batch
	local numSamples = 0
	for i=1,#classes do
		numSamples = numSamples + testClassList[i]:nElement()
	end
	local numBatches = math.ceil(numSamples/batchSize)

	local batchSizes = {}
	for i=1,#classes do
		local roundFlag = 0
		local batchSum = 0
		batchSizes[i] = {}
		for j=1,numBatches-1 do
			if roundFlag == 0 then
				batchSizes[i][j] = math.floor(testClassList[i]:nElement()/numBatches)
			else
				batchSizes[i][j] = math.ceil(testClassList[i]:nElement()/numBatches)
			end
			
			batchSum = batchSum + batchSizes[i][j]
			
			if j*testClassList[i]:nElement()/numBatches > batchSum then
				roundFlag = 1
			else
				roundFlag = 0
			end
		end
		batchSizes[i][numBatches] = testClassList[i]:nElement() - batchSum
	end

	local sampleSum = {}
	for i=1,#classes do
		sampleSum[i] = 0
	end

	for i=1,numBatches do
		-- split testClassList into batches
		sampleList = {}
		for j=1,#classes do
			sampleList[j] = testClassList[j][{{sampleSum[j] + 1, sampleSum[j] + batchSizes[j][i]}}]
			sampleSum[j] = sampleSum[j] + batchSizes[j][i]
		end

		local testset = getSample(classes, sampleList, imagePaths)
		if cudaFlag == 1 then
			testset.data = testset.data:cuda()
			testset.label = testset.label:cuda()
		end

		for j=1,testset:size() do
			local groundtruth = testset.label[j]
			local prediction = net:forward(testset.data[j])
			local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
			if groundtruth == indices[1] then
				class_performance[groundtruth] = class_performance[groundtruth] + 1
				correct = correct + 1
			end
			class_number[groundtruth] = class_number[groundtruth] + 1
		end
	end

	for i=1,#classes do
		print(classes[i], 100*class_performance[i]/class_number[i] .. ' %')
	end
	print(100*correct/numSamples .. ' %')
end
