# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imutils
from utils import *

IS_CURVE_THRESHOLD = 15000  # Threshold how many points in box to be considered as valid curve
# window settings
WINDOW_HEIGHT = 64  # Break image into 16 vertical layers since image height is 1024
MARGIN = 100  # How much to slide left and right for searching
EXPLORE_STEP = 80  # Second Exploration step (width = 2 * step)
EXPLORE_BOX_WIDTH = EXPLORE_STEP * 2
MINPIX = 50 # Set minimum number of pixels found to recenter window
COLORS = [[255, 0, 0], [255, 128, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [128, 0, 255]]
NUMBER_OF_SLICE = 3


# OUTPUT:
# 1. All Non-zero point for each curve
# 2. Curve fit function for each curve
#

class LaneProcessor(object):

    def __init__(self, binary_image, color_image, Minv):
        self.original_image = binary_image
        self.rotated = 0
        self.Minv = Minv
        self.color_image = color_image
        self.image = self.rotate_if_needed()
        nonzero = self.image.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        self.raw_curve_before_rotate_back = []  # all the candidate curve before purging and rotate back

        self.nonzero_for_curves = []

        self.position_for_curves = []

        self.coeff_for_curves = []
        self.is_solid_for_curves = []
        self.color_for_curves = []
        self.valid_points_for_curves = []

    def rotate_if_needed(self):
        rotated_image = imutils.rotate_bound(self.original_image, 90)
        _, slice_count_ori = calculate_sample_base_from_histogram_for_image(self.original_image, 1)
        _, slice_count_rotated = calculate_sample_base_from_histogram_for_image(rotated_image, 1)

        print(slice_count_ori, slice_count_rotated)

        if slice_count_ori[0] < slice_count_rotated[0] * 1.5:
            self.rotated = 1
            self.color_image = imutils.rotate_bound(self.color_image, 90)
            return rotated_image

        return self.original_image.copy()

    def draw_indicators(self):
        out_img = np.dstack((self.original_image, self.original_image, self.original_image)) * 255
        #out_img = np.dstack((self.image, self.image, self.image)) * 255

        #for i in range(len(self.coeff_for_curves)):

            #self.draw_the_curve(self.coeff_for_curves[i], is_solid=self.is_solid_for_curves[i])

        # for i in range(len(self.raw_curve_before_rotate_back)):
        #     self.draw_the_curve(self.raw_curve_before_rotate_back[i])
        # for i in range(len(self.nonzero_for_curves)):
        #     out_img[self.nonzeroy[self.nonzero_for_curves[i]], self.nonzerox[self.nonzero_for_curves[i]]] = COLORS[(i % len(COLORS))]

#         plt.imshow(out_img)
#         plt.xlim(0, 1280)
#         plt.ylim(1024, 0)
#         plt.show()

    def get_coeffs(self): ##call this
        result = []
        for coeff in self.coeff_for_curves:
            coeff = list(coeff)
            coeff.append(self.rotated)
            result.append(coeff)
        return result


    def find_curve_function(self):
        histogram = np.sum(self.image[int(self.image.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines

        number_of_slice = NUMBER_OF_SLICE
        slice_bases, _ = calculate_sample_base_from_histogram_for_image(self.image, number_of_slice)
        # Choose the number of sliding windows
        nwindows = np.int(self.image.shape[0] / WINDOW_HEIGHT)

        # Current positions to be updated for each window
        slicex_current = slice_bases #.copy()

        # Create empty lists to receive left and right lane pixel indices
        slice_inds = []
        for _ in slice_bases:
            slice_inds.append([])
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_y_low = self.image.shape[0] - (window + 1) * WINDOW_HEIGHT
            win_y_high = self.image.shape[0] - window * WINDOW_HEIGHT
            win_slices_low_highx = []
            for slice_pointx in slicex_current:
                win_slices_low_highx.append((slice_pointx - MARGIN, slice_pointx + MARGIN))

            # Identify the nonzero pixels in x and y within the window
            for i, low_high_pairx in enumerate(win_slices_low_highx):  # !!
                good_slice_inds = \
                ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= low_high_pairx[0]) & (
                    self.nonzerox < low_high_pairx[1])).nonzero()[0]
                slice_inds[i].append(good_slice_inds)

                #print(len(slice_inds[i]), "haha"+str(i))
                if len(good_slice_inds) > MINPIX:
                    slicex_current[i] = np.int(np.mean(self.nonzerox[good_slice_inds]))

        # Concatenate the arrays of indices
        #print(slice_inds)
        for index in range(len(slice_inds)):  # !!
            slice_inds[index] = np.concatenate(slice_inds[index])

        slice_inds = [x for x in slice_inds if len(x) > IS_CURVE_THRESHOLD]

        self.nonzero_for_curves.extend(slice_inds)

        ##### To decide which one has most inds

        inbound_slice = np.array([len(x) for x in slice_inds])

        fittest_curve_number = np.argmax(inbound_slice)

        fit_coefficients = []
        # Extract left and right line pixel positions
        for slice in slice_inds:  # !!
            slicex = self.nonzerox[slice]
            slicey = self.nonzeroy[slice]
            slice_fit = np.polyfit(slicey, slicex, 2)
            fit_coefficients.append(slice_fit)
            self.raw_curve_before_rotate_back.append(slice_fit)

        #self.coeff_for_curves.extend(fit_coefficients)
        all_coefficients = []

        the_fittest_curve_coeff = fit_coefficients[fittest_curve_number]
        valid_starts = self.explore_whole_image_with_coeff(the_fittest_curve_coeff)
        filtered_starts = eliminate_duplication(valid_starts, step=EXPLORE_STEP)

        #print(valid_starts)
        exploring_coefficients = []
        for start_point in filtered_starts:
            coeff_temp = []
            coeff_temp.append(the_fittest_curve_coeff[0])
            coeff_temp.append(the_fittest_curve_coeff[1])
            coeff_temp.append(start_point)
            exploring_coefficients.append(coeff_temp)

        # RECALCULATE THE FINAL COEFF
        for coeff in exploring_coefficients:
            coeff_new, inds_new = self.calculate_new_curve_by_old_curve_boxing(coeff)
            #all_coefficients.append(coeff_new)
            #self.coeff_for_curves.append(coeff_new)
            self.raw_curve_before_rotate_back.append(coeff_new)
            self.nonzero_for_curves.append(np.concatenate(inds_new))


        # Fit a second order polynomial to each

        ploty = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        slice_fitxs = []
        for fit_coeff in fit_coefficients:
            slice_fitx = fit_coeff[0] * ploty ** 2 + fit_coeff[1] * ploty + fit_coeff[2]
            slice_fitxs.append(slice_fitx)

    def explore_whole_image_with_coeff(self, coeff):
        step = EXPLORE_STEP
        VALID_CURVE_THRESH = 8000
        valid_exploring_pointx = []
        start_exploring_pointx = int(coeff[2])
        temp_point = start_exploring_pointx
        while temp_point > - (self.image.shape[1] / 2):
            temp_point -= step  # Negative exploring start
        while temp_point < self.image.shape[1]:
            exploring_centers = self.curve_reboxing(coeff[0], coeff[1], temp_point) #?
            point_in_centers = []
            for center in exploring_centers:
                point_in_centers.append(len(self.points_in_box(center))) #?

                # Here we can check if it is dot / solid lane
            total_points = sum(point_in_centers)
            if total_points > VALID_CURVE_THRESH:
                valid_exploring_pointx.append((temp_point, total_points))
            temp_point += step
        return valid_exploring_pointx

    def curve_reboxing(self, coeff2, coeff1, coeff0):  # given coeffients redraw the boxes
        plotys = [int(x * WINDOW_HEIGHT) for x in range(0, self.image.shape[0] // WINDOW_HEIGHT + 1)]
        plotxs = [int(coeff2 * ploty ** 2 + coeff1 * ploty + coeff0) for ploty in plotys]
        return zip(plotxs, plotys)

    def points_in_box(self, box_center):
        low_high_pair = box_center_to_low_high_pair(box_center, WINDOW_HEIGHT, EXPLORE_BOX_WIDTH)
        good_slice_inds = \
        ((self.nonzeroy >= low_high_pair[0][1]) & (self.nonzeroy < low_high_pair[1][1]) & (self.nonzerox >= low_high_pair[0][0]) & (
            self.nonzerox < low_high_pair[1][0])).nonzero()[0]
        return good_slice_inds

    def draw_the_curve(self, coeff, is_solid=True):  # if not rotated, use y as plot line, otherwise x.
        plot_line = np.linspace(0, self.image.shape[self.rotated] - 1, self.image.shape[self.rotated])
        other_dimension = coeff[0] * plot_line ** 2 + coeff[1] * plot_line + coeff[2]
        color_of_the_curve = 'yellow' if is_solid else 'blue'
        plt.plot(other_dimension, plot_line, color=color_of_the_curve)

    def calculate_new_curve_by_old_curve_boxing(self, coeff, drawing=True, valid=False):
        box_centers = self.curve_reboxing(coeff[0], coeff[1], coeff[2])
        if valid:
            box_centers = [x for x in box_centers if is_window_inside_image(self.image, x)]
        inds = []
        for center in box_centers:
            low_high_pair = box_center_to_low_high_pair(center, WINDOW_HEIGHT, MARGIN * 2)
            inds.append(self.points_in_box(center))
            if drawing:
                cv2.rectangle(self.image, low_high_pair[0], low_high_pair[1], (0, 255, 0), 2)

        cat_inds = np.concatenate(inds)
        # if drawing:
        #     image[nonzeroy[inds], nonzerox[inds]] = COLORS[random.randint(0,len(COLORS)-1)]
        curve_x = self.nonzerox[cat_inds]
        curve_y = self.nonzeroy[cat_inds]
        curve_fit = np.polyfit(curve_y, curve_x, 2)
        # if drawing:
        #     image[curve_y, curve_x] = COLORS[random.randint(0, len(COLORS) - 1)]


        # if self.rotated:
        #     points = list(zip(curve_x, curve_y))
        #     rotated_back = rotate_back(self.image, points)
        #     curve_x = rotated_back[0]
        #     curve_y = rotated_back[1]


            #print(curve_x)
            # if drawing:
            #     image[curve_y, curve_x] = COLORS[random.randint(0,len(COLORS)-1)]

        return curve_fit, inds

    def process_from_raw_coeff_to_result(self):
        # print(len(self.raw_curve_before_rotate_back))
        # print('++++++++++++++')
        # print(len(self.nonzero_for_curves))

        good_one_indices = filter_good_inds(self.nonzero_for_curves)
        self.raw_curve_before_rotate_back = [self.raw_curve_before_rotate_back[i] for i in good_one_indices]

        for coeff in self.raw_curve_before_rotate_back:
            coeff_new, inds = self.calculate_new_curve_by_old_curve_boxing(coeff, valid=True)
            nb_point_in_each_window = [len(x) for x in inds]



            # OUTPUT POSITION




            # OUTPUT IS SOLID LANE
            self.is_solid_for_curves.append(is_solid_lane(nb_point_in_each_window))


            # OUTPUT CURVE
            cat_inds = np.concatenate(inds)
            #print('.............')
            self.color_for_curves.append(self.is_yellow_by_inds(cat_inds))
            #print('.............')
            curve_x = self.nonzerox[cat_inds]
            curve_y = self.nonzeroy[cat_inds]
            if self.rotated:
                points = list(zip(curve_x, curve_y))
                rotated_back = rotate_back(self.image, points)
                curve_x = rotated_back[0]
                curve_y = rotated_back[1]
                coeff_new = np.polyfit(curve_y, curve_x, 2)
            self.coeff_for_curves.append(coeff_new)

        for i, curve in enumerate(self.coeff_for_curves):
            plot_line = np.linspace(0, self.image.shape[0] - 1, num=50)
            other_dimension = curve[0] * plot_line ** 2 + curve[1] * plot_line + curve[2]
            #points = zip(plot_line, other_dimension)
            # if self.rotated:
            #     points = zip(other_dimension, plot_line)
            # else:
            #     points = zip(other_dimension, plot_line)
            points = zip(other_dimension, plot_line)

            valid_points = [x for x in points if is_window_inside_image(self.image, x)]
            self.valid_points_for_curves.append(valid_points)

            valid_x = [x[0] for x in valid_points]
            valid_y = [y[1] for y in valid_points]
            # print(valid_x)
            # print(valid_y)
            # print('-------------')
            #plt.plot(valid_x, valid_y, color='yellow' if self.color_for_curves[i] else 'white')


    def is_yellow_by_inds(self, inds):
        curve_x = self.nonzerox[inds]
        curve_y = self.nonzeroy[inds]
        color_score = 0.0
        total_valid_number = 0.0
        #print(self.color_image.shape)
        for point in zip(curve_y, curve_x):
            #print(self.color_image[point])
            # if(point[0] >= 1024):
            #     continue
            score = color_of_point(self.color_image[point])
            color_score += score
            if not score == 0.0:
                total_valid_number += 1
        # print(color_score, "COLOR SCORE")
        # print(len(inds))
        # print(total_valid_number)
        if color_score > total_valid_number / 10:
            return True
        return False

    def full_process(self):
        self.find_curve_function()
        self.process_from_raw_coeff_to_result()
        return self.output()  #[self.valid_points_for_curves, self.color_for_curves, self.is_solid_for_curves]

    def output(self):
        result = []
        for curve_id in range(len(self.valid_points_for_curves)):
            points = self.valid_points_for_curves[curve_id]
            points = np.array([(x,y,1) for (x,y) in points])
            points = points.T
            points = self.Minv.dot(points).T
            points = sorted(points, key=lambda x: -x[1])

            point_arr = []
            ret_string = ""
            for point in points:
                point_arr.append(int(point[0]))
                point_arr.append(int(point[1]))

            if self.color_for_curves[curve_id]:
                ret_string += "黄色"
            else:
                ret_string += "白色"

            if self.is_solid_for_curves[curve_id]:
                ret_string += "实线"
            else:
                ret_string += "虚线"
            result.append([ret_string, point_arr])
        return result
