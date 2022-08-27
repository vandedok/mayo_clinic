import multiprocessing as mproc
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


def get_thumbnail(slide_path, thumbnail_max_size=500):
    slide_PIL = Image.open(slide_path)
    slide_size = slide_PIL.size
    max_size = max(slide_size)
    ratio = thumbnail_max_size / max_size
    thumbnail_size = tuple(int(x * ratio) for x in slide_size)
    slide_PIL.thumbnail(thumbnail_size)
    return slide_PIL

def get_rescaled(path, scale_factor = 0.1):
    with rasterio.open(path) as slide:
        # resample data to target shape
        arr = slide.read(
            out_shape=(
                slide.count,
                int(slide.height * scale_factor),
                int(slide.width * scale_factor)
            ),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
    return arr



class SlideManager():
    
    def __init__(
        self, 
        window_yx=(256,256), 
        tile_fg_criterion = 0.1,
        tile_bg_brightness = 0.1,
        slide_thresh_params = {
            "local": True,
            "block_size_factor": 0.05,
            "offset":10,
            "erode_n_it": 2,
            "erode_kernel": np.ones((5,5))
            
        },
    ):
        

        self.window_yx = window_yx
        self.tile_fg_criterion = tile_fg_criterion
        self.slide_path = None
        self.slide_thresh_params = slide_thresh_params
        self.tile_bg_brightness_int = np.round(tile_bg_brightness * 255).astype(int)
        
    def new_slide(self, slide_path, foreground_map_path=None, downscaled_path=None,
        n_cpus=1):
        
        self.slide_path = slide_path
        with rasterio.open(slide_path) as slide:
            self.size_yx = (slide.height, slide.width)
        self.grid_size_yx = tuple(int(np.ceil(s / w)) for s, w in zip(self.size_yx, self.window_yx))
        
        if downscaled_path:
            self.downscaled = np.load(foreground_map_path, allow_pickle=False)
        else:
            self.downscaled = self.get_downscaled_slide(slide_path)
        
        if foreground_map_path:
            self.foreground_map = np.load(foreground_map_path, allow_pickle=False)
        else:
            self.foreground_map = self.detect_foreground(self.downscaled, n_cpus=n_cpus)

        
    def get_region_borders(self, grid_y, grid_x):
        grid_yx = (grid_y, grid_x)
        start_yx = tuple(g * w for w, g in zip(self.window_yx, grid_yx))
        stop_yx = tuple(st + w  if st + w < s else s for w, s, st in zip(self.window_yx, self.size_yx, start_yx))
        for d in range(2):
            if start_yx[d] > self.size_yx[d]:
                raise ValueError("Problem with grid ids, grid_y: %i, grid_x %i"%grid_yx)
        return start_yx, stop_yx
    
    def get_region(self, grid_y, grid_x):
        
        if self.slide_path is None:
            raise Exception("No slide is provided.")
            
        start_yx, stop_yx = self.get_region_borders(grid_y, grid_x)
        with rasterio.open(self.slide_path) as slide:
            region = slide.read(
                window = (
                    (start_yx[0], start_yx[0]+self.window_yx[0]),
                    (start_yx[1], start_yx[1]+self.window_yx[1])
                )
            )
            
        return region
    
    def get_multiple_regions(self, grid_yxs, n_cpus=1):
        '''
        grid_yxs: list of (x,y)
        '''
        with mproc.Pool(n_cpus) as pool:
            regions = pool.starmap(self.get_region, grid_yxs)
        regions = np.stack(regions)
        
        return regions
    
    def show_region(self, grid_y, grid_x):
        region = self.get_region(grid_y, grid_x)
        plt.imshow(np.moveaxis(region, 0,-1))
        plt.show()
        
    def get_downscaled_slide(self, path):
        
        with rasterio.open(path) as slide:
        # resample data to target shape
            downscaled = slide.read(
            out_shape=(
                    slide.count,
                    int(self.grid_size_yx[0]),
                int(self.grid_size_yx[1])
                ),
                resampling=rasterio.enums.Resampling.bilinear
            )

        return downscaled
    
    def region_is_fg(self, grid_y, grid_x):
        region = self.get_region(grid_y, grid_x)
#         _, fg_map = self.thresholding(region, **self.tile_thresh_params)

        fg_map = np.max(region, axis=0) < self.tile_bg_brightness_int
        fg_fraction = np.sum(fg_map)/ fg_map.size
        if fg_fraction > self.tile_fg_criterion:
            return True
        else:
            return False        
    
    def refine_foreground(self, foreground_map, n_cpus=1):
        fg_ids = np.where(foreground_map)
        yxs = np.vstack(fg_ids).T

        with mproc.Pool(n_cpus) as pool:
            is_fg = pool.starmap(self.region_is_fg, yxs)
        yxs_refined = yxs[is_fg]    
        refined_map = np.zeros_like(foreground_map)
        refined_map[(yxs_refined[:,0], yxs_refined[:,1])] = 1
        refined_map = cv2.dilate(refined_map, kernel=np.ones((2,2)))
        return refined_map
    
    def thresholding(
        self, 
        img,
        block_size_factor=0.05, 
        erode_kernel = np.ones((5,5)), 
        offset = 10, 
        erode_n_it = 1
    ):

        img = np.moveaxis(img, 0, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
        thresh_block_size = int(max(img.shape[0]*block_size_factor, img.shape[1]*block_size_factor))
        if thresh_block_size % 2 == 0:
            thresh_block_size += 1
        if thresh_block_size == 1:
            thresh_block_size = 3
        thresh = cv2.adaptiveThreshold(
            img, 
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 
            thresh_block_size, 
            offset
        )
     
        img = (img < thresh).astype(np.uint8)  
        foreground_map = cv2.dilate(img, kernel=erode_kernel, iterations=erode_n_it)
        return thresh, foreground_map
    
    def detect_foreground(
        self, 
        downscaled,
        n_cpus=1,
    ):
        
          

        _, foreground_map = self.thresholding(downscaled, **self.slide_thresh_params)

        # Exclude border patches from foreground to avoid regions of different size.
        # Can be done more elegantly with padding, but probably don't worth it now.
        # foreground_map = np.ones(downscaled.shape[1:3]).astype(np.uint8)
        foreground_map = self.refine_foreground(foreground_map, n_cpus=n_cpus)

        foreground_map[:,-1] = 0
        foreground_map[-1,:] = 0

        return foreground_map
    