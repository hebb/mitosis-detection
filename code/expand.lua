function expand(net)
	convCount = 0
	poolCount = 0
	for i=1,net:size() do
		if torch.typename(net:get(i)) == 'nn.SpatialConvolution' then
			convCount = convCount + 1
			nInputPlane = net:get(i).nInputPlane
			nOutputPlane = net:get(i).nOutputPlane
			kW = net:get(i).kW
			kH = net:get(i).kH
			dilationW = 2^(convCount-1)
			dilationH = 2^(convCount-1)
			net:insert(nn.SpatialDilatedConvolution(nInputPlane,nOutputPlane,kW,kH,1,1,0,0,dilationW,dilationH), i+1)
			net:get(i+1).weight = net:get(i).weight
			net:get(i+1).bias = net:get(i).bias
			net:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.SpatialMaxPooling' then
			poolCount = poolCount + 1
			kW = net:get(i).kW
			kH = net:get(i).kH
			dilationW = 2^(poolCount-1)
			dilationH = 2^(poolCount-1)
			net:insert(nn.SpatialDilatedMaxPooling(kW,kH,1,1,0,0,dilationW,dilationH), i+1)
			net:get(i+1).weight = net:get(i).weight
			net:get(i+1).bias = net:get(i).bias
			net:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.View' then
			net:insert(nn.Identity(),i+1)
			net:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.Linear' then
			convCount = convCount + 1
			j = i - 1
			while true do
				if torch.typename(net:get(j)) == 'nn.SpatialDilatedConvolution' then
					break
				end
				j = j - 1
			end
			local nInputPlane = net:get(j).nOutputPlane

			local outputSize = net:get(i).weight:size(1)
			local inputSize = net:get(i).weight:size(2)

			local nOutputPlane = outputSize
			kW = torch.sqrt(inputSize/nInputPlane)
			kH = kW
			dilationW = 2^(convCount-1)
			dilationH = 2^(convCount-1)

			net:insert(nn.SpatialDilatedConvolution(nInputPlane,nOutputPlane,kW,kH,1,1,0,0,dilationW,dilationH), i+1)
			net:get(i+1).weight = net:get(i).weight:resize(nOutputPlane,nInputPlane,kH,kW)
			net:get(i+1).bias = net:get(i).bias
			net:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.LogSoftMax' then
			net:insert(nn.SpatialLogSoftMax(), i+1)
			net:remove(i)
		end
	end
	for i=net:size(),1,-1 do
		if torch.typename(net:get(i)) == 'nn.Identity' then
			net:remove(i)
		end
	end

	return net
end
