import numpy as np
import torch
class Cutout:
    def __init__(self, n_holes=1, length=16): self.n_holes=n_holes; self.length=length
    def __call__(self, img):
        h,w=img.size(1),img.size(2); mask=np.ones((h,w),np.float32)
        for _ in range(self.n_holes):
            y,x=np.random.randint(h),np.random.randint(w); d=self.length//2
            y1,y2=np.clip([y-d,y+d],0,h); x1,x2=np.clip([x-d,x+d],0,w); mask[y1:y2,x1:x2]=0.
        return img*torch.from_numpy(mask).to(img.device,img.dtype).expand_as(img)
