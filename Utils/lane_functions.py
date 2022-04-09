import numpy as np
import cv2
import math

def calculate_curvature(y_values, l1_y, l1_x, l2_y, l2_x):
    y_per_pix = 32/720
    x_per_pix = 3.7/1280
    coeff1 = np.polyfit(l1_y*y_per_pix, l1_x*x_per_pix, 2)
    coeff2 = np.polyfit(l2_y*y_per_pix, l2_x*x_per_pix, 2)
    point_y = np.max(y_values)*y_per_pix
    curves = [coeff1, coeff2]
    curvature = []
    for i in curves:
        a = math.pow((1 + np.square(2*i[0]*point_y*y_per_pix+ i[1])), 3/2)
        b = np.abs(2*i[0]) 
        curvature.append(a/b)
    if coeff1[0]<0:
        turn = 'right'
    elif coeff1[0]>0:
        turn = 'left'
    else:
        turn = 'straight'
    return curvature, turn
    

def find_lane_pixels(warped_img, p1, p2,w, degree ):
    windows = 10
    window_height = int(warped_img.shape[0]/windows)
    window_width = int((1/12) * w)  # Window width is +/- margin
    min_no_pix = int((1/24) * w)
    lane_pixels = np.where(warped_img!=0)
    lane_pixels_y = lane_pixels[0]
    lane_pixels_x = lane_pixels[1]
    lane1_inds = []
    lane2_inds = []
    for i in range(windows):
        #y coordinates for window
        y_max = warped_img.shape[0] - i*window_height
        y_min = warped_img.shape[0] - (i+1)*window_height
        #x coordinates of window for lane 1
        x1_min = p1 - window_width
        x1_max = p1 + window_width
        #x coordinates of window for lane 2
        x2_min = p2 - window_width
        x2_max = p2 + window_width
        # extracting non zero pixels in the windows
        lane1_pix = ((lane_pixels_y >= y_min) & (lane_pixels_y < y_max) & (lane_pixels_x >= x1_min) & (lane_pixels_x < x1_max)).nonzero()[0]
        # print(lane1_pix)
        # print(good_left_inds.shape)
        lane2_pix = ((lane_pixels_y >= y_min) & (lane_pixels_y < y_max) & (lane_pixels_x >= x2_min) & (lane_pixels_x < x2_max)).nonzero()[0]
                                                            
        # Append these indices to the lists
        lane1_inds.append(lane1_pix)
        lane2_inds.append(lane2_pix)
            
        #recenter next window on mean position
        if len(lane1_pix) > min_no_pix:
            p1 = int(np.mean(lane_pixels_x[lane1_pix]))
        if len(lane2_pix) > min_no_pix:        
            p2 = int(np.mean(lane_pixels_x[lane2_pix]))
                        
        # Concatenate the arrays of indices
    total_lane1_inds = np.concatenate(lane1_inds)
    total_lane2_inds = np.concatenate(lane2_inds)

    lane1_x = lane_pixels_x[total_lane1_inds]
    lane1_y = lane_pixels_y[total_lane1_inds] 
    lane2_x = lane_pixels_x[total_lane2_inds] 
    lane2_y = lane_pixels_y[total_lane2_inds]
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    y = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    if(degree == 2):

        lane1_curve = np.polyfit(lane1_y, lane1_x, 2)
        lane2_curve = np.polyfit(lane2_y, lane2_x, 2) 
            
        # Create the x and y values to plot on the image  
        
        lane1_fitx = lane1_curve[0]*y**2 + lane1_curve[1]*y + lane1_curve[2]
        lane2_fitx = lane2_curve[0]*y**2 + lane2_curve[1]*y + lane2_curve[2]
        curvature, t = calculate_curvature(y, lane1_y, lane1_x,lane2_y, lane2_x )
    else:
        lane1_curve = np.polyfit(lane1_y, lane1_x, 1)
        lane2_curve = np.polyfit(lane2_y, lane2_x, 1) 
            
        # Create the x and y values to plot on the image  
        lane1_fitx = lane1_curve[0]*y + lane1_curve[1]
        lane2_fitx = lane2_curve[0]*y + lane2_curve[1]
        curvature = []
        t = 'straight'

    
    # Generate an image to visualize the result
    out_img = np.dstack((warped_img, warped_img, (warped_img))) * 255
            
    # Add color to the left line pixels and right line pixels
    out_img[lane_pixels_y[total_lane1_inds], lane_pixels_x[total_lane1_inds]] = [0, 255, 0]
    out_img[lane_pixels_y[total_lane2_inds], lane_pixels_x[total_lane2_inds]] = [0, 0, 255]

    return out_img, lane1_fitx, lane2_fitx, y, curvature, t

def overlay_lane_lines(warped_frame, image, left_pts, right_pts, pts_y, inv_matrix, f):
    color_warp = np.zeros_like(warped_frame).astype(np.uint8)         
    pts_left = np.array([np.transpose(np.vstack([left_pts, pts_y]))], dtype='int')
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_pts, pts_y])))], dtype = 'int')
    pts = np.hstack((pts_left, pts_right))
         
    # Draw lane on the warped image
    cv2.fillPoly(color_warp, [pts], (255,0, 0))
    if not f:
        cv2.polylines(color_warp, [pts_left], False, (0,0,255), 20)
        cv2.polylines(color_warp, [pts_right], False, (0,255,0), 20)
    else:
        cv2.polylines(color_warp, [pts_left], False, (0,255,0), 20)
        cv2.polylines(color_warp, [pts_right], False, (0,0,255), 20)
    newwarp = cv2.warpPerspective(color_warp, inv_matrix, (image.shape[1], image.shape[0]))
     
    result = cv2.addWeighted(image, 0.8, newwarp, 1, 0)   
 
    return result