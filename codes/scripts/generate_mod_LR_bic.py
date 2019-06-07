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
    sourcedir = 'D:\\tmp\\BasicSR\\data_samples\\DIV2K_valid_HR_save'
    savedir = 'D:\\tmp\\BasicSR\\data_samples\\DIV2K_valid_HR_save_mod'

    saveLRpath = os.path.join(savedir, 'LR', 'x'+str(up_scale))
    
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    
    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover '+str(saveLRpath))
    
    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    pbar = tqdm(total=len(filepaths))

    def resize_func(filename):
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
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)

        pbar.update(1)
    
    # prepare data with augementation

    # # for iter
    # for filename in filepaths:
    #     resize_func(filename)
    
    # ThreadPool
    with ThreadPoolExecutor() as executor:
        executor.map(resize_func, filepaths)


if __name__ == "__main__":
    generate_mod_LR_bic()
