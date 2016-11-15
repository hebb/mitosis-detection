function test(net, classes, testClassList, imagePaths, batchSize)
	net:evaluate()

	class_performance = {0, 0}
	correct = 0
	class_number = {0, 0}

	-- compute size of each batch
	batchSizes, numBatches, numSamples = getBatchSizes(classes, testClassList, batchSize)

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
		if cudaFlag then
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
