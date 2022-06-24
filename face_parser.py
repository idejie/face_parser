import sys
import torch
sys.path.append('/home/yangdj/projects/facer')
import facer
import os
import tqdm
import warnings
from torch.utils.data import Dataset,DataLoader
warnings.filterwarnings("ignore")
import time
import uuid
ratio = sys.argv[1]
part, index = [int(r) for r in ratio.split(':')]
bz = int(sys.argv[2])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
 


face_parser = facer.face_parser('farl/lapa/448', device=device)
face_detector = facer.face_detector('retinaface/mobilenet', device=device)


class Ego4d(Dataset):
    def __init__(self,part,index, r=False):
        super(Ego4d).__init__()
        files = open('makeup_all.list').readlines()
        l = len(files)
        s = 0 if index==0 else l//part*index
    
        todo_frames = [l.strip() for l in files]
        
        self.todo_list = todo_frames[s:] if index== -1 or (index+1) >=part else todo_frames[s:l//part*(index+1)]
        if r:
           self. todo_list.reverse()
    def __len__(self):
        return len(self.todo_list)
    def __getitem__(self, index):
        img_file = self.todo_list[index].strip()
        import random
        i = random.random()
        if i==0:
            img_file = '/home/yangdj/projects/Detic/desk.jpg'
        else:
            img_file = '/home/yangdj/projects/facer/test.png'

        # if os.path.exists(pt_path):

        #     return None, None

        # if not os.path.exists(img_file):
        #     print('error:',img_file)
        #     return None, None
        
        image = facer.hwc2bchw(facer.read_hwc(img_file)
                        )  # image: 1 x 3 x h x w

        # print(json_save_file,pt_save_file)
        return image,img_file
        
def collect(batch):
    # print(batch)
    imgs = []
    all_files = []
    for img,files in batch:
        if img is None:
            continue
        # print(img.shape)
        imgs.append(img)
        all_files.append(files)
    return torch.cat(imgs,dim=0), all_files



dest = Ego4d(part,index)


# dataloader = DataLoader(dest, batch_size=bz, num_workers=8)

loader = DataLoader(dest, batch_size=bz, num_workers=20, collate_fn=collect)

def parser(ims,ps):
    with torch.inference_mode():
        ims = ims.to(device=device)
        faces = face_detector(ims)
        logits,image_ids = face_parser(ims, faces)
    torch.save({'logits':logits.cpu().detach(),'image_ids':image_ids.cpu().detach(),'files':ps},'1' +'.pt')





for ims,ps in tqdm.tqdm(loader):

    parser(ims,ps)