import cv2
import numpy as np
import math


def binarize(image_name):
    im_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    #(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 30
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('bw'+image_name.split('.')[0]+'.jpg', im_bw)
    #return im_bw

# binarize('TSD-Lane-00068-00074.BEVLane.png')


def rotate_back(image, points):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # perform the rotation
    M_inv = cv2.getRotationMatrix2D(center, 90, 1)
    # add ones
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # transform points
    transformed_points = M_inv.dot(points_ones.T)
    transformed_points = np.add(transformed_points, (np.hstack([ones*128, ones*-128]).T))
    return transformed_points.astype(int)


def box_center_to_low_high_pair(center, box_height, box_width):
    return ((center[0] - box_width // 2, center[1] - box_height // 2),
            (center[0] + box_width // 2, center[1] + box_height // 2))


# [(-39, 26331), (41, 55017), (121, 28686), (361, 16224), (441, 12429), (1001, 29957), (1081, 35176)]
def eliminate_duplication(arr, step=80):
    arr_dict = dict(arr)
    for pair in arr:
        if arr_dict.get(pair[0], None):
            for i in [-2, -1, 1, 2]:
                if arr_dict.get(pair[0]+i*step, None):
                    if arr_dict[pair[0]+i*step] >= arr_dict[pair[0]]:
                        arr_dict.pop(pair[0], None)
                        break
                    else:
                        arr_dict.pop(pair[0]+i*step, None)
    return list(arr_dict)


def calculate_sample_base_from_histogram_for_image(image, nb_slices=3):
    histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)
    slice_bases = []
    slice_counts = []
    slice_range = int(histogram.shape[0] / nb_slices)
    for i in range(nb_slices):
        slice_bases.append(np.argmax(histogram[i * slice_range: (i + 1) * slice_range]) + i * slice_range)
        slice_counts.append(np.max(histogram[i * slice_range: (i + 1) * slice_range]) + i * slice_range)
    return slice_bases, slice_counts


def is_solid_lane(nb_point_in_boxes):
    box_numbers = len(nb_point_in_boxes)
    point_threshold = np.mean(nb_point_in_boxes) / 4
    #print(point_threshold)
    is_box_avove_thresh = [(1 if (x - point_threshold > 0) else -1) for x in nb_point_in_boxes]
    smoothed_consolidated_lane = __consolidate_arr(__smooth_the_dot(__consolidate_arr(is_box_avove_thresh)))
    #print(smoothed_consolidated_lane)
    #print(smoothed_consolidated_lane)
    if len(smoothed_consolidated_lane) > 3:
        return False
    if len(smoothed_consolidated_lane) == 0:
        #print("ERROR WRONG SOLID JUDGE")
        return True
    if sum(smoothed_consolidated_lane) > box_numbers / 3.0: #len = 1,2,3
        return True
    else:
        return False
    return True #len == 1


def is_window_inside_image(image, window_center):
    if (0 <= window_center[0] <= image.shape[1]) and (0 <= window_center[1] <= image.shape[0]):
        return True
    return False


    # print(is_box_avove_thresh)
    #
    # consolidated = __consolidate_arr(is_box_avove_thresh)
    # print(consolidated)
    #
    # smoothed = __smooth_the_dot(consolidated)
    # print(smoothed)
    # print(__consolidate_arr(smoothed))

    # Intuition: beginning and ending 0 are not counting, then if

    # [0,0,0, 1, 1,1,1, 0,0,1]
    # [1,0,1,0,1] -> [1,1,1,1,1] [1,0,0,0,1]
    # [1,1,0,1,1] -> [1,1,1,1,]


def __consolidate_arr(arr):
    if not arr:
        return arr
    result = []
    temp = arr[0]
    for i in range(1, len(arr)):
        if arr[i] * temp < 0:
            result.append(temp)
            temp = arr[i]
        else:
            temp += arr[i]
    result.append(temp)
    return result


def __smooth_the_dot(arr):
    result = []
    for i in range(len(arr)):
        if arr[i] == -1:
            if 0 < i < len(arr) - 1:
                if arr[i-1] + arr[i+1] >= 4:
                    result.append(1) # from -1 to 1
        result.append(arr[i])
    return result


def filter_good_inds(inds_arr):
    if not inds_arr:
        return []
    result = []
    result.append(0)
    for j in range(1, len(inds_arr)):
        if len(inds_arr[j]) < 8000:
            continue
        for i in range(len(result)):
            comparison = __compare_two_inds(inds_arr[j], inds_arr[result[i]])
            if comparison == 0:
                if i == len(result) - 1:
                    result.append(j)
            if comparison == -1:
                break
            if comparison == 1:
                result[i] = j
                break
    if len(inds_arr[result[0]]) < 8000:
        result.pop(0)
    return result



def __compare_two_inds(inds1, inds2):
    if float(len(set(inds1) - set(inds2))) / float(len(inds1)) < 0.2 or float(len(set(inds2) - set(inds1))) / float(len(inds2)) < 0.2:
        return 1 if len(set(inds1)) > len(set(inds2)) else -1
    return 0


    union = set(inds1).union(inds2)
    intersect = set(inds1).intersection(inds2)
    ratio = float(len(union))/float(len(intersect))
    if ratio > 1.5:
        return 0

def color_of_point(point_rgb):
    if (float(point_rgb[0]) + float(point_rgb[1])) < (256. / 255.):
        return 0.0

        #print()
    return (175. - point_rgb[2] * 255.) / 175.




"""
[[ 1152.   -93.]
 [ 1152.    47.]
 [  952.   -23.]
 [  937.   -23.]]

"""
