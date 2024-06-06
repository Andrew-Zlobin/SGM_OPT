import argparse
import sys
import time as t

import cv2
import numpy as np


from multiprocessing import Process, Manager




class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):

        self.direction = direction
        self.name = name



N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self):

        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):

        self.max_disparity = max_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize


def load_images(left_name, right_name, parameters):

    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
    return left, right


def get_indices(offset, dim, direction, height):

    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, parameters):

    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths, return_value=[None], return_value_index=0):


    print("start process", return_value_index)

    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1


    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    dawn = t.time()
    for path in paths.effective_paths:
        # print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
    print("finish process", return_value_index)
        
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return_value[return_value_index] = aggregation_volume
    return aggregation_volume

def cross_based_cost_agregation(cost_volume, parameters, paths, return_value=[None], return_value_index=0):
    pass


def compute_costs_by_improved_census(left, right, parameters, save_images):
    
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width = left.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right improved census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = np.int64(0)

            # compute census window 
            I = cwidth
            M = 3 * I
            T = 20
            center_pixel = left[y, x]
            while I < M:
                I_offset = int (I / 2)
                if y - I_offset < 0 or y + I_offset + 1 >= height or x - I_offset < 0 or x + I_offset + 1 >= width:
                    I -= 2
                    break
                X = (left[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)] == center_pixel).all(axis=-1).sum()
                if X < T:
                    break
                I+=2
                T+=1
            I_offset = int (I / 2)

            array_to_delete = left[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)].ravel()
            mask = np.logical_or(array_to_delete == array_to_delete.max(), array_to_delete == array_to_delete.min())
            center_pixel = np.ma.masked_array(array_to_delete, mask = mask).mean()
            # print(center_pixel)
            #left[y, x]
            
            reference = np.full(shape=(I, I), fill_value=center_pixel, dtype=np.int64)
            image = left[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (I_offset, I_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_img_census[y, x] = np.uint8(left_census)
            left_census_values[y, x] = left_census

            right_census = np.int64(0)

            # compute census window 
            I = cwidth
            M = 5 * I
            T = 20
            center_pixel = left[y, x]
            while I < M:
                I_offset = int (I / 2)
                if y - I_offset < 0 or y + I_offset + 1 >= height or x - I_offset < 0 or x + I_offset + 1 >= width:
                    I -= 2
                    break
                X = (right[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)] == center_pixel).all(axis=-1).sum()
                if X < T:
                    break
                I+=2
                T+=1
            I_offset = int (I / 2)


            # center_pixel = right[y, x]
            array_to_delete = right[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)].ravel()
            mask = np.logical_or(array_to_delete == array_to_delete.max(), array_to_delete == array_to_delete.min())
            center_pixel = np.ma.masked_array(array_to_delete, mask = mask).mean()
            reference = np.full(shape=(I, I), fill_value=center_pixel, dtype=np.int64)
            image = right[(y - I_offset):(y + I_offset + 1), (x - I_offset):(x + I_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (I_offset, I_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_img_census[y, x] = np.uint8(right_census)
            right_census_values[y, x] = right_census

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))


    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def compute_costs(left, right, parameters, save_images):

    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width = left.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = np.int64(0)
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            # print('reference = ', reference)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                            # print('True bit')
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_img_census[y, x] = np.uint8(left_census)
            left_census_values[y, x] = left_census

            right_census = np.int64(0)
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_img_census[y, x] = np.uint8(right_census)
            right_census_values[y, x] = right_census

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))
    print("left_census_values = ", np.unique(left_census_values))
    print("right_census_values = ", np.unique(right_census_values))


    if save_images:
        cv2.imwrite('left_census.png', left_img_census)
        cv2.imwrite('right_census.png', right_img_census)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume):

    print("aggregation_volume.shape = ", aggregation_volume.shape)

    volume = np.sum(aggregation_volume, axis=3)
    print("volume = ", np.unique(volume))
    print("volume.shape = ", volume.shape)
    disparity_map = np.argmin(volume, axis=2)
    print("disparity_map.shape = ", disparity_map.shape)
    print("disparity_map = ", np.unique(disparity_map))
    return disparity_map


def normalize(volume, parameters):

    return 255.0 * volume / parameters.max_disparity


def get_recall(disparity, gt, disp):

    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    gt = np.int16(gt / 255.0 * float(disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size


def sgm():

    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='cones/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='cones/disp6.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    disparity = args.disp
    save_images = args.images
    evaluation = args.eval

    dawn = t.time()

    parameters = Parameters(max_disparity=disparity, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))
    paths = Paths()

    print('\nLoading images...')
    left, right = load_images(left_name, right_name, parameters)

    print('\nStarting cost computation...')
    #compute_costs
    left_cost_volume, right_cost_volume = compute_costs_by_improved_census(left, right, parameters, save_images)
    if save_images:
        left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
        cv2.imwrite('disp_map_left_cost_volume.png', left_disparity_map)
        right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
        cv2.imwrite('disp_map_right_cost_volume.png', right_disparity_map)

    print('\nStarting left aggregation computation...')

    aggregation_volume = aggregation_volume.values()
    left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)#, aggregation_volume, 0) #aggregation_volume[0]
    # print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths) #
    # print(left_aggregation_volume)
    # print(right_aggregation_volume)
    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), parameters))
    right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), parameters))
    
    dusk = t.time()

    if save_images:
        cv2.imwrite('left_disp_map_no_post_processing.png', left_disparity_map)
        cv2.imwrite('right_disp_map_no_post_processing.png', right_disparity_map)

    

    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, parameters.bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, parameters.bsize[0])
    cv2.imwrite(f'left_{output_name}', left_disparity_map)
    cv2.imwrite(f'right_{output_name}', right_disparity_map)
    
    

    if evaluation:
        print('\nEvaluating left disparity map...')
        recall = get_recall(left_disparity_map, left_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
        print('\nEvaluating right disparity map...')
        recall = get_recall(right_disparity_map, right_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))

    
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))


def create_disparity_map(imgL,imgR,dispMap=[]):
    
    print("imgL =", np.unique(imgL))
    print("imgR =", np.unique(imgR))
    
    disparity = 64 #args.disp
    

    dawn = t.time()

    parameters = Parameters(max_disparity=disparity, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))
    paths = Paths()

    left, right = imgL,imgR

    print('\nStarting cost computation...')
    #compute_costs
    left_cost_volume, right_cost_volume = compute_costs(left, right, parameters, False) # _by_improved_census
    print("left_cost_volume =", np.unique(left_cost_volume))
    print("right_cost_volume =", np.unique(right_cost_volume))

    print('\nStarting left aggregation computation...')

    # aggregation_volume = aggregation_volume.values()
    left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)#, aggregation_volume, 0) #aggregation_volume[0]
    # print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths) #
    # print(left_aggregation_volume)
    # print(right_aggregation_volume)
    print('\nSelecting best disparities...')
    print("left_aggregation_volume =", np.unique(left_aggregation_volume))
    print("right_aggregation_volume =", np.unique(right_aggregation_volume))

    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), parameters))
    right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), parameters))
    
    dusk = t.time()


    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))
    print("left_disparity_map =", np.unique(left_disparity_map))
    print("right_disparity_map =", np.unique(right_disparity_map))


    dispMap = left_disparity_map.T
    return left_disparity_map




if __name__ == '__main__':
    sgm()
