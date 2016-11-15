function scan_old(img, net, windowWidth, windowHeight, N)
	img = img:double()

	local imageHeight = img:size(2)
	local imageWidth = img:size(3)

	local map = torch.FloatTensor(imageHeight, imageWidth):fill(0)
	local window = torch.Tensor(windowHeight, windowWidth)
	if cuda then
		window = window:cuda()
	end
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
		if cuda then
			window = window:cuda()
		end
		local out = net:forward(window)
		for j=1,numWinRows do
			r1 = (i-1)*N+1+(windowHeight-1)/2
			r2 = i*N+(windowHeight-1)/2
			c1 = (j-1)*N+1+(windowWidth-1)/2
			c2 = j*N+(windowWidth-1)/2
			map[{{r1,r2},{c1,c2}}] = math.exp(out[j][1])
			print(i,j,out[j][1])
		end
	end
	return map
end

function scan(img, net)
	dofile("padarray.lua")
	pady = 50
	padx = 50
	img = padarray(img,{0,pady,padx},'mirror')

	--normalize
	mean = {}
	stdv  = {}
	for i=1,3 do
		mean[i] = img[{{i}, {}, {}  }]:mean()
		img[{{i}, {}, {}  }]:add(-mean[i])

		stdv[i] = img[{{i}, {}, {}  }]:std()
		img[{{i}, {}, {}  }]:div(stdv[i])
	end
	local out = net:forward(img)

	map = torch.exp(out[{{1},{},{}}])
	map = torch.squeeze(map)

	--pad with zeros to match the size to the input size
	pady = (img:size(2) - out:size(2))/2
	padx = (img:size(3) - out:size(3))/2
	--map = padarray(map,{pady, padx}, 'zero')

	return map
end
