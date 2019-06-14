import os, sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util import imresize_np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4

    # set data dir
    sourcedir = 'D:\\tmp\\BasicSR\\data_samples\\div_train\\valid_HR'
    savedir = 'D:\\tmp\\BasicSR\\data_samples\\div_train\\valid_LR'

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        
    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')][:10]
    pbar = tqdm(total=len(filepaths))

    def scale_func(filename):
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))
    
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale*height, 0:mod_scale*width,:]
        else:
            image_HR = image[0:mod_scale*height, 0:mod_scale*width]
        # LR
        image_LR = imresize_np(image_HR, 1/up_scale, True)
        cv2.imwrite(os.path.join(savedir, filename), image_LR)

        pbar.update(1)

    def resize_func(filename):
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))
        image_LR = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(savedir, filename), image_LR)

        pbar.update(1)
    
    # prepare data with augementation

    # for iter
    # for filename in filepaths:
    #     resize_func(filename)
    
    # ThreadPool
    with ThreadPoolExecutor() as executor:
        executor.map(scale_func, filepaths)


if __name__ == "__main__":
    generate_mod_LR_bic()
