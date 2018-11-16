# some code from : https://github.com/tkuanlun350/3DUnet-Tensorflow-Brats18
import random
import numpy as np

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max
	
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode='full', bounding_box = None):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if(sample_mode[i] == 'full'):
            if(bounding_box):
                x0 = bounding_box[i*2]; x1 = bounding_box[i*2 + 1]
            else:
                x0 = 0; x1 = input_shape[i]
        else:
            if(bounding_box):
                x0 = bounding_box[i*2] + int(output_shape[i]/2)   
                x1 = bounding_box[i*2+1] - int(output_shape[i]/2)   
            else:
                x0 = int(output_shape[i]/2)   
                x1 = input_shape[i] - x0
        if(x1 <= x0):
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center
	
def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume 

def corp_and_roi(img, mask, roi_size):
	# corp
	margin = 5 # small padding value
	original_shape = img.shape[0:3]
	# corp corresponding img[..., 0]
	bbmin, bbmax = get_none_zero_region(img[... , 0], margin)
	mask = crop_ND_volume_with_bounding_box(mask, bbmin, bbmax)
	
	#get center point after corp
	center_point = get_random_roi_sampling_center(mask.shape, roi_size, "full", None)
	mask = extract_roi_from_volume(mask, center_point, roi_size, fill = 'zero')
	
	#deal img with 4 channels
	img_ = np.zeros(roi_size+ (4,))
	for mod in range(img.shape[3]):
		img__ = crop_ND_volume_with_bounding_box(img[... , mod], bbmin, bbmax)
		img__ = extract_roi_from_volume(img__, center_point, roi_size, fill = 'zero')
		img_[... , mod] = img__

	return img_, mask
	
	
def eval_slice_patches(img, patches_size):
	# input: (d, w, h, channel)
	sub_images = []
	center_points = []
	
	if len(img.shape)==4:
		D, H, W, nch = img.shape
	else:
		D, H, W= img.shape
		nch = 1
	
	input_center = [int(D/2), int(H/2), int(W/2)]
	
	#slice along depth
	for center_slice in range(int(patches_size[0]/2), D + int(patches_size[0]/2), patches_size[0]):
		center_slice = min(center_slice, D - int(patches_size[0]/2))
		sub_image = []
		temp_input_center = [center_slice, input_center[1], input_center[2]]
		center_points.append(temp_input_center)
		
		if len(img.shape)==4:
			for chn in range(nch):
				sub_image_ch = extract_roi_from_volume(
						img[..., chn], temp_input_center, patches_size, fill="zero")
				sub_image.append(sub_image_ch)
		else:
			sub_image_ch = extract_roi_from_volume(
						img, temp_input_center, patches_size, fill="zero")
			sub_image.append(sub_image_ch)
		
		sub_image = np.asanyarray(sub_image, np.float32)
		#sub_image = np.transpose(sub_image, [1, 2, 3, 0])
		sub_images.append(sub_image)

	return sub_images, center_points
	
def eval_merge_patches(imgs, center_points, original_size):
	temp_prob1 = np.zeros(original_size)
	
	for ix, cp in enumerate(center_points):
		nch = imgs[ix].shape[0]
		for chn in range(nch):
			temp_prob1[chn] = set_roi_to_volume(temp_prob1[chn], cp, imgs[ix][chn, ...])
	
	return temp_prob1
	
	
def slice_center_points(img, patches_size):
	# input: (d, w, h, channel)
	
	center_points = []
	
	if len(img.shape)==4:
		D, H, W, nch = img.shape
	else:
		D, H, W= img.shape
		nch = 1
	
	input_center = [int(D/2), int(H/2), int(W/2)]
	
	for depth in range(int(patches_size[0]/2), D + int(patches_size[0]/2), patches_size[0]):
		depth = min(depth, D - int(patches_size[0]/2))
		
		for height in range(int(patches_size[1]/2), H + int(patches_size[1]/2), patches_size[1]):
			height = min(height, H - int(patches_size[1]/2))
			
			for width in range(int(patches_size[2]/2), W + int(patches_size[2]/2), patches_size[2]):
				width = min(width, W - int(patches_size[2]/2))
				
				temp_input_center = [depth, height, width]
				center_points.append(temp_input_center)
				
	return center_points

def eval_slice_patches_cps(img, patches_size, center_points):
	# input: (d, w, h, channel)
	sub_images = []
	
	if len(img.shape)==4:
		D, H, W, nch = img.shape
	else:
		D, H, W= img.shape
		nch = 1
	
	for slice_center in center_points:
		sub_image = []
		if len(img.shape)==4:
			for chn in range(nch):
				sub_image_ch = extract_roi_from_volume(
						img[..., chn], slice_center, patches_size, fill="zero")
				sub_image.append(sub_image_ch)
		else:
			sub_image_ch = extract_roi_from_volume(
						img, slice_center, patches_size, fill="zero")
			sub_image.append(sub_image_ch)
		
		sub_image = np.asanyarray(sub_image, np.float32)
		sub_images.append(sub_image)

	return sub_images