require 'torch';
require 'sys';
require 'image';
local dir = require 'pl.dir';
local ffi = require 'ffi';

function getImagePaths(folder)
	-- obtain list of image files
	local classes = {}
	local classPaths = {}
	local dirs = dir.getdirectories(folder);
	for k,dirpath in ipairs(dirs) do
		local class = paths.basename(dirpath)
		table.insert(classes, class)
		table.insert(classPaths, dirpath)
	end

	-- define command-line tools, try your best to maintain OSX compatibility
	local wc = 'wc'
	local cut = 'cut'
	local find = 'find'
	if ffi.os == 'OSX' then
		wc = 'gwc'
		cut = 'gcut'
		find = 'gfind'
	end

	-- options for the GNU find command
	local extensionList	 = {'jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
	local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
	for i=2,#extensionList do
		findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
	end

	-- find the image path names
	local imagePaths = torch.CharTensor()	-- path to each image in dataset
	local imageClass = torch.LongTensor()	-- class index of each image (class index in self.classes)
	local classList = {}			-- index of imageList to each image of a particular class

	-- create file listing the paths to every image
	local classFindFiles = {}
	for i=1,#classes do
		classFindFiles[i] = os.tmpname()
	end
	local combinedFindList = os.tmpname()

	local tmpfile = os.tmpname()
	local tmphandle = assert(io.open(tmpfile, 'w'))
	for i,class in ipairs(classes) do
		local command = find .. ' "' .. classPaths[i] .. '" ' .. findOptions .. ' >>"' .. classFindFiles[i] .. '" \n'
		tmphandle:write(command)
	end
	io.close(tmphandle)
	os.execute('bash ' .. tmpfile)
	os.execute('rm -f ' .. tmpfile)

	local tmpfile = os.tmpname()
	local tmphandle = assert(io.open(tmpfile, 'w'))
	-- concat all finds to a single large file in the order of self.classes
	for i=1,#classes do
		local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
		tmphandle:write(command)
	end
	io.close(tmphandle)
	os.execute('bash ' .. tmpfile)
	os.execute('rm -f ' .. tmpfile)

	local maxPathLength = tonumber(sys.fexecute(wc .. " -L '" .. combinedFindList .. "' |" .. cut .. " -f1 -d' '")) + 1
	local length = tonumber(sys.fexecute(wc .. " -l '" .. combinedFindList .. "' |" .. cut .. " -f1 -d' '"))

	imagePaths:resize(length, maxPathLength):fill(0)
	local s_data = imagePaths:data()
	for line in io.lines(combinedFindList) do
		ffi.copy(s_data, line)
		s_data = s_data + maxPathLength
	end
	numSamples = imagePaths:size(1)
	print(numSamples ..  ' samples found.')

	imageClass:resize(numSamples)
	local runningIndex = 0
	for i=1,#classes do
		local length = tonumber(sys.fexecute(wc .. " -l '" .. classFindFiles[i] .. "' |" .. cut .. " -f1 -d' '"))
		classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
		imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
		runningIndex = runningIndex + length
	end

	local tmpfilelistall = ''
	for i=1,#(classFindFiles) do
		tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
		if i % 1000 == 0 then
			os.execute('rm -f ' .. tmpfilelistall)
			tmpfilelistall = ''
		end
	end
	os.execute('rm -f '  .. tmpfilelistall)
	os.execute('rm -f "' .. combinedFindList .. '"')

	return classes, classList, imagePaths
end

function getSample(classes, sampleList, imagePaths)
	dataTable = {}
	scalarTable = {}
	N = 0
	for i=1,#classes do
		for j=1,sampleList[i]:nElement() do
			local imgpath = ffi.string(torch.data(imagePaths[sampleList[i][j]]))
			out = image.load(imgpath, 3, 'float')
			table.insert(dataTable, out)
			table.insert(scalarTable, i)
			N = N + 1
		end
	end
	data = torch.Tensor(N, 3, 101, 101)
	scalarLabels = torch.LongTensor(N):fill(-1111)
	for i=1,#dataTable do
		data[i]:copy(dataTable[i])
		scalarLabels[i] = scalarTable[i]
	end
	dataset = {}
	dataset.data = data
	dataset.label = scalarLabels

	setmetatable(dataset,
		{__index = function(t, i) 
						return {t.data[i], t.label[i]}
					end}
	);

	function dataset:size() 
		return self.data:size(1) 
	end

	-- data normalization
	mean = {}
	stdv  = {}
	for i=1,3 do
		mean[i] = dataset.data[{ {}, {i}, {}, {}  }]:mean()
		dataset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
		
		stdv[i] = dataset.data[{ {}, {i}, {}, {}  }]:std()
		if stdv[i] ~= 0 then
			dataset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
		end
	end

	return dataset
end

function getRandomSample(classes, batchSize, classList, imagePaths)
	dataTable = {}
	scalarTable = {}
	N = 0
	for i=1,#classes do
		for j=1,batchSize do
			local index = math.max(1, math.ceil(torch.uniform() * classList[i]:nElement()))
			local imgpath = ffi.string(torch.data(imagePaths[classList[i][index]]))
			out = image.load(imgpath, 3, 'float')
			table.insert(dataTable, out)
			table.insert(scalarTable, i)
			N = N + 1
		end
	end
	data = torch.Tensor(N, 3, 101, 101)
	scalarLabels = torch.LongTensor(N):fill(-1111)
	for i=1,#dataTable do
		data[i]:copy(dataTable[i])
		scalarLabels[i] = scalarTable[i]
	end
	dataset = {}
	dataset.data = data
	dataset.label = scalarLabels

	setmetatable(dataset,
		{__index = function(t, i) 
						return {t.data[i], t.label[i]}
					end}
	);

	function dataset:size() 
		return self.data:size(1) 
	end

	-- data normalization
	mean = {}
	stdv  = {}
	for i=1,3 do
		mean[i] = dataset.data[{ {}, {i}, {}, {}  }]:mean()
		dataset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
		
		stdv[i] = dataset.data[{ {}, {i}, {}, {}  }]:std()
		if stdv[i] ~= 0 then
			dataset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
		end
	end

	return dataset
end

function getBatchSizes(classes, classList, batchSize)
	local numSamples = 0
	for i=1,#classes do
		numSamples = numSamples + classList[i]:nElement()
	end
	local numBatches = math.ceil(numSamples/batchSize)

	local batchSizes = {}
	for i=1,#classes do
		local roundFlag = 0
		local batchSum = 0
		batchSizes[i] = {}
		for j=1,numBatches-1 do
			if roundFlag == 0 then
				batchSizes[i][j] = math.floor(classList[i]:nElement()/numBatches)
			else
				batchSizes[i][j] = math.ceil(classList[i]:nElement()/numBatches)
			end
			
			batchSum = batchSum + batchSizes[i][j]
			
			if j*classList[i]:nElement()/numBatches > batchSum then
				roundFlag = 1
			else
				roundFlag = 0
			end
		end
		batchSizes[i][numBatches] = classList[i]:nElement() - batchSum
	end
	
	return batchSizes, numBatches, numSamples
end
