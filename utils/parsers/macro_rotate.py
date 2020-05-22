import cv2
import errno
import imutils
import numpy as np
import os
import pandas as pd
import sys

DIMENSION = 700

data_dir       = '../../data/'
resize_dir     = os.path.join(data_dir, 'resize-inpaint')
resize_df_path = os.path.join(data_dir, 'resize-inpaint_meta')

output_dir     = os.path.join(data_dir, 'macro-angles')
output_df_path = os.path.join(data_dir, 'macro-angles_meta')


def get_sorted_filenames_to_parse(in_df, out_df):
    # The format of parsed images is <original_fname_root>-<integer>.jpg so we
    # just need to get the unique set of <original_fname_root>s and compare that
    # to the full set of original filenames to figure out which files need to be
    # parsed still
    def get_source_fname(fname):
        return os.path.splitext(fname)[0].split('-deg')[0] + '.jpg'
    parsed_files = set(get_source_fname(f) for f in out_df['filename'])
    unparsed_files = set(in_df['filename']) - parsed_files
    return sorted(list(unparsed_files))

def get_rotated_coords(x, y, angle):
    # Negate to rotate clockwise, matching the rotation direction of imutils
    rads = np.deg2rad(-angle)
    c, s = np.cos(rads), np.sin(rads)
    x, y = np.dot(np.array(((c, -s), (s, c))), np.array((x, y)))
    # This gives us negative values, however. Move back into frame
    if angle in [270, 180]:
        x += DIMENSION
    if angle in [90, 180]:
        y += DIMENSION
    return int(x), int(y)

def update_dataframe(out_df, img_info, angle, fname):
    data = dict(img_info)
    data['angle'] = angle
    data['filename'] = fname

    if angle != 0:
        # Update the values for width, height, x and y based on the rotation
        if angle in [90, 270]:
            height = data['width']
            data['width'] = data['height']
            data['height'] = height

        data['x'], data['y'] = get_rotated_coords(data['x'], data['y'], angle)

    # Values need to be in lists to create a DataFrame
    for k, v in data.items():
        data[k] = [v]

    out_df = out_df.append(pd.DataFrame(data), ignore_index=True)
    out_df.to_pickle(output_df_path)
    return out_df


# =============================== Start of Script ==============================

if not os.path.exists(resize_df_path):
    print('No images to rotate')
    sys.exit(0)

input_df = pd.read_pickle(resize_df_path)
input_cols = input_df.columns

try:
    output_df = pd.read_pickle(output_df_path)
except OSError as e:
    if e.errno != errno.ENOENT:
        raise
    output_df = pd.DataFrame(
        columns=['filename', 'x', 'y', 'width', 'height', 'angle'])

unseen_fnames = get_sorted_filenames_to_parse(input_df, output_df)
unseen_df = input_df[input_df['filename'].isin(unseen_fnames)]

# Perform 0, 90, 180, and 270 degree rotations on all of the image augmentations
for index, row in unseen_df.iterrows():
    print(row.filename)
    image_path = os.path.join(resize_dir, row.filename)
    orig_img = cv2.imread(image_path)
    out_fname_temp = os.path.splitext(row.filename)[0] + '-deg{}.jpg'

    for angle in [0, 90, 180, 270]:
        rot_img = imutils.rotate(orig_img,angle)

        out_fname = out_fname_temp.format(angle)
        cv2.imwrite(os.path.join(output_dir, out_fname), rot_img)

        output_df = update_dataframe(output_df, row, angle, out_fname)
