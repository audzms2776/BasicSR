import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob


pbar = None

def main():
    global pbar
    
    """A multi-thread tool to crop sub imags."""
    input_folder = 'D:\\tmp\\data\\DIV2K_valid_HR'
    save_folder = 'D:\\tmp\\data\\valid_sample'
    crop_sz = 96
    step = crop_sz // 2
    thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    
    img_list = glob(input_folder + '/*')

    pbar = tqdm(total=len(img_list))

    with ThreadPoolExecutor() as executor:
        future_worker = {executor.submit(worker, path, save_folder, crop_sz, step, thres_sz, compression_level): path for path in img_list}
        
        proccessing_sum = 0
        
        for future in as_completed(future_worker):
            try:
                data = future.result()
                proccessing_sum += data
            except Exception as exc:
                print(exc)
            else:
                pbar.set_description('{}'.format(proccessing_sum))
            
    # for path in img_list:
    #     print(path)
    #     result = worker(path, save_folder, crop_sz, step, thres_sz, compression_level)
    
    # print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    names = img_name.split('.')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)

    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            
            cv2.imwrite(
                os.path.join(save_folder, '{}{}.{}'.format(names[0], index, names[1])),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
            
    pbar.update(1)
    return index


if __name__ == '__main__':
    main()
