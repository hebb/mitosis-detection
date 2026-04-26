# Mitosis Detection in Histopathology Images using Convolutional Neural Networks

An algorithm for detecting mitotic figures in breast cancer histology slides. This project will use Torch. The algorithm will use convolutional neural networks.

## Getting Started

Install [torch](https://github.com/torch/torch7)

Install the necessary packages.

cutorch, nn, cunn, optim

## How to Run This Code

## Torch

### train(net, criterion, classes, classList, imagePaths, batchSize, learningRate, learningRateDecay, weightDecay, momentum, maxIteration, classRatio, augment)

### test

### data

### getImagePaths

### getSample

### getRandomSample

### [batchSizes, numBatches, numSamples] getBatchSizes(classes, classList, batchSize)

Takes as input the two classes, 'classes', an array of images indices, 'classList', and the approximate batch size, 'batchsize'. 'batchSizes' is an array containing the size of each batch, where batchSize[i][j] is the batch size of the jth sample of class i. 'numSamples' is the total number of samples.

### shuffleImages(classList, classes)

Randomizes 'classList'.

### padarray(x, pad, padtype)

### [map] scan(img, net)

'map' is the output of 'net' with input 'img'. 'net' must be a dilated convolutional neural network. If it's not, use 'expand(net)' to turn it into one. 'img' is first extended by 50 pixels on all sides using mirroring. The output is as if 'img' were processed with 'net' sliding across, but the entire image is actually processed together.

### [map] scan\_old(img, net, windowWidth, windowHeight, N)

This is an old version of the function which is no longer used. Its replacement is much more efficient.

'map' is the output of 'net' with input 'img', using sliding windows of size 'windowWidth x windowHeight', spaced by 'N' pixels (to save time).

### [dataset] randRotateMirror(dataset)

Randomly rotates and flips each image in 'dataset'. 50% chance of mirroring and 25% chance each of rotation by 0, 90, 180, or 270 degrees.

### [outNet] convnet2autoencoder(inNet)

Transforms the convolutional neural network, 'inNet', into the stacked convolutional autoencoder, 'outNet'.

### [net2] autoencoder2convnet(net1, net2)

Copies the parameters of the stacked convolutional autoencoder, 'net1', to the convolutional neural network, 'net2'.

### [net] expand(net)

Turns the convolutional neural network 'net' into a dilated convolutional neural network, so that the network can process the entire image at once, which is more efficient than actually scanning across the image.

## Matlab

### [Inorm H E] normalizeStaining(I, Io, beta, alpha, HERef, maxCRef)

Takes, as input, the RGB input image, 'I', and separates into two stain channels, producing the normalized image, 'Inorm', the hemotoxylin image, 'H', and the eosin image, 'E'.

### stainSeparation(folder)

Performs the stain separation for the entire folder, 'folder'. Every file with the '.bmp' extension is given as input to the function 'normalizeStaining' with default values for the optional arguments. Three output images are produced for each.

### rgb2he(folder)

### [b] isPos(X, Y, M, d)

Returns a boolean value indicating whether the point represented by the coordinates X and Y is less than the distance d from one of the points represented by the rows of M.

### [out] highlight(image\_file, csv\_file)

Produces an image, out, similar to the input image, 'image\_file', in which the coordinates listed in 'csv\_file' are highlighted.

### bmp2png(folder)

Converts every file with the extension '.bmp' into a png image file.

### plot\_detections(imageFile, csvFile, threshold)

Produces an image file called 'detections.bmp' in which the detections listed in 'csvFile' which exceed the 'threshold' argument are marked.

### plot\_prcurve(res\_folder, gt\_folder)

Plots the precision-recall curve using the results folder, 'res\_folder', and the ground truth folder 'gt\_folder'.

### gen\_dataset

This script generates the training set of 101x101 images for the small classifier using the folder 'old\_trainset\_folder', which must be set to the location of the original 2000x2000 images training set. It expects the training set folder to contain subfolders named with the format 'A[two digit number]\_v3' and containing bmp images and csv files listing the coordinates of the mitotic figures.

This script calls the function 'add\_dataset', which operates on each image and its corresponding csv file to create the appropriate 101x101 images in the folder 'new\_trainset\_folder'.

### [len\_M, num\_true] add\_dataset(image\_file, csv\_file, new\_trainset\_folder)

Creates the 101x101 image training set for the small classifier for just the image 'image\_file' using the coordinates of the mitotic figures in 'csv\_file', and places them in 'new\_trainset\_folder.

'len\_M' is the number of mitotic figures in the image while 'num\_true' is the number of training images created.
