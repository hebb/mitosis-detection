function randRotateMirror(dataset)
	for i=1,dataset:size() do
		theta = math.pi/2*torch.random(0,3)
		sample = dataset[i][1]
		sample = sample:double()
		sample = image.rotate(sample,theta)
		if torch.rand(1,2) == 1 then
			sample = image.hflip(sample)
		end
		dataset[i][1] = sample
	end

	return dataset
end				
