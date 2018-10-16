	
def PrePro_1(input):
	# scale to 0 ~ 1
	input =  input.astype('float32')
	input /= 255.

	if(len(input.shape)<3): 
		input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1)
		
	return input
	
def color_preprocessing_CIFAR10(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test
	
