function getImagePaths(folder)
	dirs = dir.getdirectories(folder);

	imagePaths = {}
	for i,dirpath in ipairs(dirs) do
		command = 'find ' .. dirpath .. ' -iname "*.png"'
		res = sys.execute(command)
		for str in string.gmatch(res, "([^\n]+)") do
			table.insert(imagePaths, str)
		end
	end
	return imagePaths
end
