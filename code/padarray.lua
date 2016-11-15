local function dimnarrow(x,sz,pad,dim)
    local xn = x
    for i=1,x:dim() do
        if i > dim then
            xn = xn:narrow(i,pad[i]+1,sz[i])
        end
    end
    return xn
end

local function padzero(x,pad)
    local sz = x:size()
    for i=1,x:dim() do 
      sz[i] = sz[i]+pad[i]*2 
    end
    local xx = x.new(sz):zero()
    local xn = dimnarrow(xx,x:size(),pad,-1)
    xn:copy(x)
    return xx
end

local function padmirror(x,pad)
    local xx = padzero(x,pad)
    local sz  = xx:size()
    for i=1,x:dim() do
        local xxn = dimnarrow(xx,x:size(),pad,i)
        for j=1,pad[i] do
            xxn:select(i,j):copy(xxn:select(i,pad[i]*2-j+1))
            xxn:select(i,sz[i]-j+1):copy(xxn:select(i,sz[i]-pad[i]*2+j))
        end
    end
    return xx
end

function padarray(x,pad,padtype)
-- Example usage:  img = padarray(img,{0, pady, padx},'zero')
    if x:dim() ~= #pad then
        error('number of dimensions of Input should match number of padding sizes')
    end
    if padtype == 'zero' then return padzero(x,pad) end
    if padtype == 'mirror' then return padmirror(x,pad) end
    error('unknown paddtype ' .. padtype)
end
