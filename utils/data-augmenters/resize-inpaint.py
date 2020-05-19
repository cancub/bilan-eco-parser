import cv2
import errno
import numpy as np
import os
import pandas as pd
import sys

DIMENSION = 700
data_dir       = '../../data/'
input_dir      = os.path.join(data_dir, 'original')
output_dir     = os.path.join(data_dir, 'resize-inpaint')
input_df_path  = os.path.join(data_dir, 'original_meta')
output_df_path = os.path.join(data_dir, 'resize-inpaint_meta')

def get_dataframes():
    # Load the dataframe for the original images. If it's not there, we can't
    # actually parse anything, so let the exception get thrown
    input_df = pd.read_pickle(input_df_path)

    # Attempt to load the dataframe for the output images. This might be the
    # first time that we're running this script. If so, no dataframe exists yet,
    # thus we make it
    try:
        output_df = pd.read_pickle(output_df_path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        output_df = pd.DataFrame(
            columns=['filename', 'x', 'y', 'width', 'height'])
        output_df = output_df.astype(
            dict.fromkeys(['x', 'y', 'width', 'height'], np.int8))

    return input_df, output_df

def update_output_df(df, new_fname, new_details):
    df = df.append(
        pd.DataFrame({
            'filename': [new_fname],
            'x'       : [new_details['x']],
            'y'       : [new_details['y']],
            'width'   : [new_details['width']],
            'height'  : [new_details['height']],
        }),
        ignore_index=True)

    df.to_pickle(output_df_path)

    return df

def check_all_images_accounted_for(filenames, df):
    assert(len(set(filenames) - set(df['filename'])) == 0)

def get_sorted_filenames_to_parse(in_df, out_df):
    # The format of parsed images is <original_fname_root>-<integer>.jpg so we
    # just need to get the unique set of <original_fname_root>s and compare that
    # to the full set of original filenames to figure out which files need to be
    # parsed still
    def get_source_fname(fname):
        return os.path.splitext(fname)[0].split('-')[0] + '.jpg'
    parsed_files = set(get_source_fname(f) for f in out_df['filename'])
    unparsed_files = set(in_df['filename']) - parsed_files
    return sorted(list(unparsed_files))

def resize_and_crop(orig_im, details):
    # Build up the return dictionary
    new_details = dict.fromkeys(['x', 'y', 'width', 'height'])

    # Randomly resize in the range of 1x to close to the maximum that we can
    # scale while still having the screen fully in view
    MAX_SCALE = (DIMENSION/1.2) / int(details.width)
    SCALE = np.random.rand() * (MAX_SCALE - 1) + 1
    resized_dim = int(DIMENSION * SCALE)
    resized = cv2.resize(im, (resized_dim, resized_dim))

    # Collect the scaled details
    new_details['width'] = resized_width = int(details.width * SCALE)
    new_details['height'] = resized_height = int(details.height * SCALE)
    orig_centre = np.array([int(details.x), int(details.y)])
    resized_centre = (orig_centre * SCALE).astype(np.uint16)

    # Select a random DIMENSIONxDIMENSION area of the resized image, making
    # sure that we:
    #   a) don't go past the edges of the scaled image
    #   b) keep the entirety of the scale in view
    max_displacement = resized_dim - DIMENSION
    half_width  = resized_width/2
    half_height = resized_height/2
    MAX_X = min(resized_centre[0] - half_width, max_displacement)
    MAX_Y = min(resized_centre[1] - half_height, max_displacement)
    MIN_X = max(resized_centre[0] + half_width - DIMENSION, 0)
    MIN_Y = max(resized_centre[1] + half_height - DIMENSION, 0)

    SHIFT_X = int(np.random.rand() * (MAX_X - MIN_X) + MIN_X)
    SHIFT_Y = int(np.random.rand() * (MAX_Y - MIN_Y) + MIN_Y)

    # Determine the new location of the centre of the screen
    shifted_centre = resized_centre - [SHIFT_X, SHIFT_Y]
    new_details['x'], new_details['y'] = shifted_centre

    half_dims    = [half_width, half_height]
    bottom_right = shifted_centre + half_dims
    top_left     = shifted_centre - half_dims

    # Bottom right is on or within the expected right and bottom bounds
    assert(np.max(bottom_right - DIMENSION) <= 0)
    # Top right is on or greater than the expected left and top bounds
    assert(np.min(top_left) >= 0)

    # Get the subsection
    cropped = resized[SHIFT_Y:(SHIFT_Y + DIMENSION),
                      SHIFT_X:(SHIFT_X + DIMENSION), :]

    return cropped, new_details

def inpaint(orig_im, x, y, width, height):
    '''
    Starting from a few pixels to the left and above the screen, inpaint a
    random portion of the the screen. We do this by picking three values:
        1. depth
        2. random point between leftmost and rightmost part of screen
        3. ditto
    We then inpaint in the area described by (min(2,3), top), (max(2,3), depth)
    '''
    # Give some space so that the inpainter has a harder time filling in the
    # gaps with a clean sheet of red
    BUFFER  = 0.1 * height
    START_Y = max(int(y - height/2 - BUFFER), 0)
    # Not including the buffer area, cover between 10% and 90% of the height of
    # the screen
    MIN_Y   = int(START_Y + height*0.1 + (0 if START_Y == 0 else BUFFER))
    MAX_Y   = int(START_Y + height*0.9 + (0 if START_Y == 0 else BUFFER))
    MIN_X   = max(x - width/2 - BUFFER, 0)
    MAX_X   = min(MIN_X + width + BUFFER, orig_im.shape[1])

    depth = int(np.random.rand() * (MAX_Y - MIN_Y) + MIN_Y)
    def get_x():
        return int(np.random.rand() * (MAX_X - MIN_X) + MIN_X)

    # Get the two x points and make sure that the width of the resulting mask
    # will be at least 20% of the screen
    x_pts = sorted([get_x() for i in range(2)])
    while (x_pts[1] - x_pts[0]) <= 0.2 * width:
        x_pts = sorted([get_x() for i in range(2)])

    # Make the mask for the inpainter
    mask = np.zeros(orig_im.shape[:2], np.uint8)
    mask[START_Y:depth+1, x_pts[0]:x_pts[1]+1] = 255
    inpainted = cv2.inpaint(orig_im, mask, 3, cv2.INPAINT_NS)
    return inpainted

def show_img(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =================================== Script ===================================

# Get the dataframes
input_df, output_df = get_dataframes()

# Get the images that we have already parsed
parsed_filenames = os.listdir(output_dir)

# Do a quick sanity check to see that all of the images that are in the output
# directory are represented in the output dataframe
check_all_images_accounted_for(parsed_filenames, output_df)

for fname in get_sorted_filenames_to_parse(input_df, output_df):
    # Load in the next unparsed image
    fpath = os.path.join(input_dir, fname)
    im = cv2.imread(fpath)
    print(fname)

    # Load details from the original dataframe (the image ID is the same as the
    # index in the dataframe)
    im_details = input_df[input_df['filename'] == fname]
    # Build the bottom right point because we're going to reference it
    point = (im_details.x, im_details.y)

    # Use the parent name for the template of the filename that all the new
    # files will use
    filename_tmp = os.path.splitext(fname)[0] + '-{}.jpg'

    # We want to get 16 images out of the original image
    for i in range(16):
        new_fname = filename_tmp.format(str(i).zfill(8))

        # Zoom and crop
        im_cropped, new_details = resize_and_crop(im, im_details)

        # Inpaint a region of the screen
        final_im = inpaint(im_cropped, **new_details)

        # Save the new image
        new_filepath = os.path.join(output_dir, new_fname)
        cv2.imwrite(new_filepath, final_im)

        # Record its information in the output dataframe
        output_df = update_output_df(output_df, new_fname, new_details)
