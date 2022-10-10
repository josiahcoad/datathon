
import numpy as np
import itertools
from typing import List
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TileEdges:
    top: List[int]
    bottom: List[int]
    right: List[int]
    left: List[int]


@dataclass(frozen=True)
class TiledImage:
    top_left: List[List[int]]
    top_right: List[List[int]]
    bottom_right: List[List[int]]
    bottom_left: List[List[int]]

    @property
    def array(self):
        # 0, 1
        # 2, 3
        return np.array([self.top_left, self.top_right, self.bottom_left, self.bottom_right])

    def __getitem__(self, index):
        return self.array[index]


def array_sim(src, dst):
    return (src & dst).sum()


def score_permutation(top_left_tile, top_right_tile, bottom_left_tile, bottom_right_tile):
    pairs = [
        (top_left_tile.right, top_right_tile.left),
        (top_right_tile.bottom, bottom_right_tile.top),
        (bottom_right_tile.left, bottom_left_tile.right),
        (bottom_left_tile.top, top_left_tile.bottom),
    ]
    return sum(array_sim(src, dst) for src, dst in pairs)


def get_arrangement_similarities(tiles: List[TileEdges]):
    arrangement_sims = []
    for arrangement in itertools.permutations([0, 1, 2, 3]):
        top_left, top_right, bottom_left, bottom_right = np.array(tiles)[list(arrangement)]
        arrangement_sim = score_permutation(top_left, top_right, bottom_left, bottom_right)
        arrangement_sims.append((arrangement, arrangement_sim))
    # sort by max sim
    return list(reversed(sorted(arrangement_sims, key=lambda tup: tup[1])))


def get_best_arrangement(arrangement_sims):
    return arrangement_sims[0][0]


def tile_img(img):
    # we expect the img to be square and even number pixels... but just in case it isn't,
    # lets make it this way.
    width, height, _ = img.shape
    min_dim = min(width, height)
    if min_dim % 2 != 0:
        min_dim -= 1
    img = img[:min_dim, :min_dim]
    
    half_w = width // 2
    half_h = height // 2
    return TiledImage(
        top_left=img[:half_w,:half_h,:],
        top_right=img[:half_w,half_h:,:],
        bottom_left=img[half_w:,:half_h,:],
        bottom_right=img[half_w:,half_h:,:],
    )


def extract_edges(tile):
    return TileEdges(
        top=tile[0, :],
        bottom=tile[-1, :],
        left=tile[:, 0],
        right=tile[:, -1],
    )


def stitch_tiles(tiles: List[TiledImage]):
    top = np.concatenate((tiles[0], tiles[1]), axis=1)
    bottom = np.concatenate((tiles[2], tiles[3]), axis=1)
    return np.concatenate((top, bottom), axis=0)


def read_image(filename):
    return cv2.imread(filename)


def get_contour(img):
    tiles = tile_img(img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detemine contour thresholds
    v = np.median(gray_img)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    def get_contour_(img):
        # filter image
        bilateral = cv2.bilateralFilter(img, 15, 75, 75)

        # apply the Canny Edge filter
        contour = cv2.Canny(bilateral, lower_thresh, upper_thresh)
        contour = contour // 255
        return contour

    return TiledImage(
        top_left=get_contour_(tiles.top_left),
        top_right=get_contour_(tiles.top_right),
        bottom_left=get_contour_(tiles.bottom_left),
        bottom_right=get_contour_(tiles.bottom_right),
    )


def load_image(filename):
    img = read_image(filename)
    return get_contour(img)


def predictor(filename=None, contour=None):
    tiles = load_image(filename) if filename else contour
    edges = [extract_edges(tiles.top_left), extract_edges(tiles.top_right),
             extract_edges(tiles.bottom_left), extract_edges(tiles.bottom_right)]

    arrangement_sims = get_arrangement_similarities(edges)

    best_arrangement = get_best_arrangement(arrangement_sims)
    return np.argsort(best_arrangement)


def show_results(img):
    color_tiles = tile_img(img)
    tiles = get_contour(img)
    edges = [extract_edges(tiles.top_left), extract_edges(tiles.top_right), extract_edges(tiles.bottom_left), extract_edges(tiles.bottom_right)]

    arrangement_sims = get_arrangement_similarities(edges)

    best_arrangement = get_best_arrangement(arrangement_sims)
    second_best = arrangement_sims[1][0]
    third_best = arrangement_sims[2][0]
    fourth_best = arrangement_sims[3][0]

    first = stitch_tiles(tiles[list(best_arrangement)])
    second = stitch_tiles(tiles[list(second_best)])
    third = stitch_tiles(tiles[list(third_best)])
    fourth = stitch_tiles(tiles[list(fourth_best)])

    color_first = stitch_tiles(color_tiles[list(best_arrangement)])
    color_second = stitch_tiles(color_tiles[list(second_best)])
    color_third = stitch_tiles(color_tiles[list(third_best)])
    color_fourth = stitch_tiles(color_tiles[list(fourth_best)])
    
    f, axarr = plt.subplots(4,2)

    axarr[0,0].set_title(f'First Guess', fontsize=10)
    axarr[0,0].imshow(color_first)
    axarr[1,0].set_title(f'Second Guess', fontsize=10)
    axarr[1,0].imshow(color_second)
    axarr[2,0].set_title(f'Third Guess', fontsize=10)
    axarr[2,0].imshow(color_third)
    axarr[3,0].set_title(f'Fourth Guess', fontsize=10)
    axarr[3,0].imshow(color_fourth)

    axarr[0,1].imshow(first)
    axarr[1,1].imshow(second)
    axarr[2,1].imshow(third)
    axarr[3,1].imshow(fourth)

    for i in range(4):
        for j in range(2):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])

    f.tight_layout()
    st.pyplot(fig=f)

# %%


# # %%
# filename = '../train/0312/00039.png'

# plt.imread(filename)
# plt.imshow(plt.imread(filename))
# plt.show()
# plt.imshow(stitch_tiles(load_image(filename)))
# show_results(filename)

# %%


import streamlit as st
from PIL import Image 

image_files = [ 'scrambled.png', 'scrambled2.png', 'scrambled3.png', 'scrambled4.png']


def load_image_file(image_file):
	img = Image.open(image_file)
	return img

st.title('Image DeScRaMbLeR! 🤪')
with st.expander('How it works'):
    st.text('''1. Split the image into four tiles
2. Remove noise from by blurring a bit
3. Apply edge detection (output is matrix of 1's and 0's, 1 if pixel is edge)
4. Extract the border pixels from each tile (an array of 1s and 0s)
5. For each of the 24 (4!) possible arrangements...
6. - compare the neighboring borders of each tile
7. - Get "border similiarity" which we calculate as
     the number of pairwise pixels which are both 1
8. Sort the possible arrangements by total border similarity''')

cols = st.columns(4)

selection = st.selectbox('Select Picture To Descramble', options=[1,2,3,4])

for i in range(4):
    cols[i].image(load_image_file(image_files[i]), f'Image {i+1}')

img = load_image_file(image_files[selection-1])
img_arr = np.array(img)
show_results(img_arr)