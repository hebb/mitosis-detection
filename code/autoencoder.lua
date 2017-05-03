function convnet2autoencoder(inNet)
	outNet = inNet:clone()

	for i=outNet:size(),1,-1 do
		if torch.typename(outNet:get(i)) == 'nn.View' then
			outNet:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.Linear' then
			outNet:remove(i)
		elseif torch.typename(net:get(i)) == 'nn.LogSoftMax' then
			outNet:remove(i)
		end
	end

	for i=outNet:size(),1,-1 do
		if torch.typename(outNet:get(i)) == 'nn.SpatialMaxPooling' then
			local pool_layer = nn.SpatialMaxPooling(2,2,2,2)
			outNet:insert(pool_layer,i+1)
			outNet:remove(i)
			outNet:add(nn.SpatialMaxUnpooling(pool_layer))
		elseif torch.typename(outNet:get(i)) == 'nn.SpatialConvolution' then
			nInputPlane = outNet:get(i).nOutputPlane
			nOutputPlane = outNet:get(i).nInputPlane
			kW = outNet:get(i).kW
			kH = outNet:get(i).kH
			outNet:add(nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH))
			outNet:add(nn.ReLU())
		end
	end

	return outNet
end

function autoencoder2convnet(net1, net2)
	-- get indices for convolution layers for net1
	convList1 = {}
	j = 1
	for i=1,net1:size() do
		if torch.typename(net1:get(i)) == 'nn.SpatialConvolution' then
			convList1[j] = i
			j = j + 1
		end
	end

	-- get indices for convolution layers for net2
	convList2 = {}
	j=1
	for i=1,net2:size() do
		if torch.typename(net2:get(i)) == 'nn.SpatialConvolution' then
			convList2[j] = i
			j = j + 1
		end
	end

	-- copy parameters from net1 to net2
	for i=1,#convList1 do
		net2:get(convList2[i]).weight = net1:get(convList1[i]).weight
		net2:get(convList2[i]).bias = net1:get(convList1[i]).bias
	end

	return net2
end
