from Utils.misc import *
from Utils.lane_functions import *

def main():
    video = 'whiteline.mp4'
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('lane1.mp4',fourcc, 15, (960, 540))

    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while True:
        ret, frame = cap.read()
        if ret == True:
            image = frame.copy()
            # image = cv2.flip(image, 1)
            hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            # mask for white lane
            white_mask = cv2.inRange(hls_img, (0, 200, 0), (255, 255,255))
            # mask for yellow lane
            yellow_mask = cv2.inRange(hls_img, (20,120,80), (45, 200, 255))
            # final mask
            mask = cv2.bitwise_or(white_mask, yellow_mask)
            # thresholding the L channel of HLS 
            _,L_thresh = cv2.threshold(hls_img[:,:,1], 128, 255, cv2.THRESH_BINARY)
            L_thresh = cv2.GaussianBlur(L_thresh, (3, 3), 0)
            # Perform sobel edge detection on the thresholded channel
            sobel_binary = sobel(L_thresh)            
            combined_binary = cv2.bitwise_or(mask, sobel_binary.astype(np.uint8))
            src_pts = np.array([[100, image.shape[0]-1], [450, image.shape[0]- 220 ], [520, image.shape[0]-220], [910,image.shape[0]-1]], np.float32)
            dst_pts = np.array([[100, image.shape[0]-1], [100, 0], [910, 0], [910, image.shape[0]-1]], np.float32)
            lane_warped, inv_matrix = perspective_transform(combined_binary, src_pts, dst_pts)
            histogram = np.sum(lane_warped[lane_warped.shape[0]//2:,:], axis=0)
            midpoint = int(histogram.shape[0]/2)
            peak1 = np.argmax(histogram[:midpoint])
            peak2 = np.argmax(histogram[midpoint:]) + midpoint
            if histogram[peak1] > histogram[peak2]:
                flag = True
            else:
                flag =False
            # print(histogram[peak1])
            # print(histogram[peak2])
            lane_line_img, ind1, ind2, y_ind,_, _= find_lane_pixels(lane_warped, peak1, peak2, image.shape[1], 1)
            img_warped, inv_matrix = perspective_transform(image, src_pts, dst_pts)
            final_lane = overlay_lane_lines(img_warped, image, ind1, ind2, y_ind, inv_matrix, flag )
            cv2.imshow('frame', final_lane) 
            video.write(final_lane)
            # plt.imshow(combined_binary, cmap = 'gray')
            # plt.plot(histogram)
            # plt.show()
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        else:
            break

    cap.release()
    video.release()

if __name__ == '__main__':
    main()