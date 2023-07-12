import numpy as np
from PIL import Image

def recover_image(img):
    if len(img.shape)==3:
        img=img.cpu().numpy().transpose(1, 2, 0)*255
    else:
        img=img.cpu().numpy().transpose(0, 2, 3, 1)*255
    return img.astype(np.uint8)
def save_images(imgs):
    height=recover_image(imgs[0])[0].shape[0]
    width=recover_image(imgs[0])[0].shape[1]
    total_width=width*len(imgs)
    
    new_im = Image.new('RGB', (total_width+len(imgs), height))
    for i,img in enumerate(imgs):
        result = Image.fromarray(recover_image(img)[0])
        new_im.paste(result, (i*width+i,0))
    return new_im

def prepare_images(listoflists,batch_size):
    res=[]
    for i in range(batch_size):
        for l in listoflists:
            res.append(l[i])

    return res
