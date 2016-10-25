function scan(img, net, windowWidth, windowHeight)
	local imageHeight = img:size(2)
	local imageWidth = img:size(3)

	local map = torch.FloatTensor(imageHeight, imageWidth):fill(0)
	local window = torch.Tensor(windowHeight, windowWidth)
	window = window:cuda()
	numWinRows = math.ceil((imageHeight-windowHeight+1)/N)
	for i=1,numWinRows do	
		window = torch.DoubleTensor(numWinRows, 3, windowHeight, windowWidth)
		for j=1,numWinRows do
			window[{{j}, {}, {}, {}}] = image.crop(img, N*(j-1), N*(i-1), N*(j-1)+windowWidth, N*(i-1)+windowHeight)
		end
		--normalize
		mean = {}
		stdv  = {}
		for i=1,3 do
			mean[i] = window[{{}, {i}, {}, {}  }]:mean()
			window[{{}, {i}, {}, {}  }]:add(-mean[i])

			stdv[i] = window[{{}, {i}, {}, {}  }]:std()
			window[{{}, {i}, {}, {}  }]:div(stdv[i])
		end
		window = window:cuda()
		local out = net:forward(window)
		for j=1,numWinRows do
			r1 = (i-1)*N+1+(windowHeight-1)/2
			r2 = i*N+(windowHeight-1)/2
			c1 = (j-1)*N+1+(windowWidth-1)/2
			c2 = j*N+(windowWidth-1)/2
			map[{{r1,r2},{c1,c2}}] = math.exp(out[j][1])
		end
	end
	return map
end
