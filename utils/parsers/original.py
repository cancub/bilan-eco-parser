import cv2
import errno
import imutils
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import sys

DIMENSION = 700

# ============================== Utility functions =============================

def manual_rotate(orig_image):
    angle = 0
    while True:
        # Show the image at a particular angle
        user_rotated = imutils.rotate(orig_image, angle)
        cv2.imshow('Rotate', user_rotated)

        # Wait for the user to specify something
        key = cv2.waitKey(0)
        if key == 27:
            # Esc
            print('exiting')
            sys.exit(0)
        elif key == 81:
            # left arrow key for small clockwise
            angle -= 0.2
        elif key == 44:
            # left triangle bracket for large clockwise
            angle -= 5
        elif key == 83:
            # right arrow key for small counter-clockwise
            angle += 0.2
        elif key == 46:
            # right triangle bracket for large counter-clockwise
            angle += 5
        elif key == 13:
            # Enter key to signify that all changes are good to go
            break

    return user_rotated

zoom_points     = []

# For the points of the screen, we select the three points that are most often
# visible and then infer the shape of the screen from them
three_corners   = [] # Order is important here: left, top, bottom-right
total_clicks    = 0
cropping_img    = None

def click_to_zoom(event, x, y, flags, param):
    global zoom_points
    # grab references to the global variables
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        zoom_points = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        zoom_points.append((x, y))

def click_to_crop(event, x, y, flags, param):
    # grab references to the global variables
    global cropping_img, three_corners, total_clicks
    # if the left mouse button was clicked, record the (x, y) coordinates and
    # display the point
    if event == cv2.EVENT_LBUTTONDOWN:
        three_corners.append((x, y))
        total_clicks += 1
        if total_clicks < 3:
            # draw a circle on the corner
            cv2.circle(cropping_img, (x,y), 2, (0, 255, 0), 2)
        else:
            # draw the inferred rectangle
            cv2.rectangle(cropping_img,
                (three_corners[0][0], three_corners[1][1]), (x, y),
                (0, 255, 0), 2)
        cv2.imshow("image", cropping_img)


def retreive_screen_shape(orig_image):
    global cropping_img, zoom_points, three_corners, total_clicks

    # load the image, clone it, and setup the mouse callback function
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_to_zoom)

    zoom_points = []

    print('Select area to zoom in')
    cv2.imshow("image", orig_image)
    if cv2.waitKey(0) == 27:
        # Esc
        print('exiting')
        sys.exit(0)

    # Now we have the range of the image to zoom to
    height = zoom_points[1][1] - zoom_points[0][1]
    width  = zoom_points[1][0] - zoom_points[0][0]

    to_zoom = orig_image[zoom_points[0][1]:zoom_points[1][1],
                         zoom_points[0][0]:zoom_points[1][0]]

    # Scale up a bit to make it easier to draw the outline
    SCALE = int(800 / width)
    zoomed_portion = cv2.resize(to_zoom, (width*SCALE, height*SCALE))

    # Load the cropped region into the global and then make a copy in case we
    # want to redraw the region
    cropping_img = zoomed_portion
    clone = zoomed_portion.copy()
    cv2.setMouseCallback("image", click_to_crop)

    # keep looping until the enter key is pressed
    print('Select screen area')
    total_clicks = 0
    three_corners = []
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", cropping_img)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord('r'):
            total_clicks = 0
            three_corners = []
            cropping_img = clone.copy()
        elif key == ord('\r'):
            break
        elif key == 27:
            # Esc
            print('exiting')
            sys.exit(0)

    top_left = (three_corners[0][0], three_corners[1][1])
    bottom_right = three_corners[2]

    scale_height = int( (bottom_right[1] - top_left[1]) / SCALE )
    scale_width  = int( (bottom_right[0] - top_left[0]) / SCALE )

    # If we scale down the position of the bottom-right screen point and then add
    # that to the top-left point of zoom_points (on the original image), we get
    # the position of the bottom-right point in the original image.
    bottom_right = (np.array(bottom_right) / SCALE).astype(np.int16) + np.array(zoom_points[0])

    # close all open windows
    cv2.destroyAllWindows()

    return bottom_right, scale_height, scale_width

def get_weight(orig_image):
    weight = ''
    while True:
        # Show the image at a particular angle
        cv2.imshow('Rotate', orig_image)

        # Wait for the user to specify something
        key = cv2.waitKey(0)
        if key == 27:
            # Esc
            print('exiting')
            sys.exit(0)
        elif key == 8:
            # backspace to fix weight
            if weight != '':
                weight = weight[:-1]
            print(weight)
        elif key in range(48, 58):
            # number key to enter weight
            weight += chr(key)
            print(weight)
        elif key == 13:
            # Enter key to signify that all changes are good to go
            break

    return int(weight)



# =================================== Script ===================================

data_dir = '../../data/'
raw_dir  = os.path.join(data_dir, 'raw')
orig_dir = os.path.join(data_dir, 'original')
df_path  = os.path.join(data_dir, 'original_meta')

try:
    df = pd.read_pickle(df_path)
except OSError as e:
    if e.errno != errno.ENOENT:
        raise
    df = pd.DataFrame(
        columns=['filename', 'weight', 'br_x', 'br_y', 'width', 'height'])

# look in original/ to figure out the next available image ID
try:
    im_id = int(os.path.splitext(sorted(os.listdir(orig_dir))[-1])[0]) + 1
except IndexError:
    im_id = 0

for fname in os.listdir(raw_dir):
    # Load in the image
    image_path = os.path.join(raw_dir, fname)

    # Get the new filename
    filename = str(im_id).zfill(8) + '.jpg'
    new_filepath = os.path.join(orig_dir, filename)

    print('{} --> {}'.format(image_path, new_filepath))

    # Scale it down to some size that'll fit the screen
    img = cv2.resize(cv2.imread(image_path), (DIMENSION, DIMENSION))

    # Perform manual rotation to bring the scale screen to rougly 0 degrees
    print('Rotate image')
    r_img = manual_rotate(img)

    # Get the value on the scale
    print('Enter weight on scale: ')
    weight = get_weight(r_img)
    cv2.destroyAllWindows()

    # Manually determine the location of the scale screen
    bottom_right, height, width = retreive_screen_shape(r_img)

    # Save the smaller image in the new directory to indicate that it has been
    # parsed
    cv2.imwrite(new_filepath, r_img)

    # Now that the new, smaller file has been created, delete the older file
    try:
        os.unlink(image_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

    # Add to dataframe with 'filename', 'br_x, 'br_y', 'width', 'height' and
    # write to "original_meta" dataframe in main data directory for reference
    df = df.append(
        pd.DataFrame({
                'filename': [filename],
                'weight': [weight],
                'br_x': [bottom_right[0]],
                'br_y': [bottom_right[1]],
                'width': [width],
                'height': [height],
            }),
        ignore_index=True)

    df.to_pickle(df_path)

    im_id += 1
    print('-'*30)
