function b = isPos(X,Y,M,d)
    for i=1:size(M,1)
        r = (X-M(i,1))^2 + (Y-M(i,2))^2;
        if r < d*d
            b = true;
            return 
        end
        b = false;
        return
    end
end