require 'torch';
require 'image';

folder = '/home/andrew/mitosis-detection'

train_folder = paths.concat(folder,'mitosis-train')
test_folder = paths.concat(folder,'mitosis-test')
trainset = paths.concat(folder,'mitosis-train.t7')
testset = paths.concat(folder,'mitosis-test.t7')

function gen_dataset(data_folder, outfile)
	true_folder = paths.concat(data_folder,'true')
	false_folder = paths.concat(data_folder,'false')

	true_files = {}
	false_files = {}

	for file in paths.files(true_folder,'.png') do
		table.insert(true_files, paths.concat(true_folder,file))
	end

	for file in paths.files(false_folder,'.png') do
		table.insert(false_files, paths.concat(false_folder,file))
	end

	data = torch.ByteTensor(#true_files+#false_files,3,101,101)
	label = torch.ByteTensor(#true_files+#false_files)

	for i,file in ipairs(true_files) do
		print(i)
		data[{{i},{},{},{}}] = image.load(file,3,'byte')
		label[i] = 1
	end

	for i,file in ipairs(false_files) do
		print(i + #true_files)
		data[{{i + #true_files},{},{},{}}] = image.load(file,3,'byte')
		label[i + #true_files] = 1
	end

	dataset = {}
	dataset.data = data
	dataset.label = label

	torch.save(outfile, dataset)
end

gen_dataset(train_folder,trainset)
gen_dataset(test_folder,testset)
