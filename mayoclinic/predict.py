import math
import numpy as np
import torch
from torch import nn

def identity(x):
    return x

class MayoPredictor(nn.Module):
    
    def __init__(
        self,
        slide_manager,
        fine_model,
        coarse_model,
        embedding_size=None,
        batch_size=1,
        n_cpus=1,
        device = torch.device("cpu"),
        progress_bar=identity
                ):
        
        super().__init__()
        self.slide_manager = slide_manager
        self.fine_model = fine_model
        self.coarse_model = coarse_model
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = self.get_embedding_size()
        self.progress_bar=progress_bar
        self.device = device
        self.fine_model.eval()
       
        
    def get_coarse_repr(self, slide_path, foreground_map_path=None, downscaled_path=None):
        self.slide_manager.new_slide(
            slide_path,
            foreground_map_path=foreground_map_path,
            downscaled_path=downscaled_path,
            n_cpus=self.n_cpus
        )
        fg_ids = np.where(self.slide_manager.foreground_map)
        yxs = np.vstack(fg_ids).T
        num_regions = yxs.shape[0]
        
        batch_size = self.batch_size
        num_batches = math.floor(num_regions / batch_size)
        
        batches_bounds = [(batch_size*i,batch_size*(i+1)) for i in range(num_batches)]
        last_batch_size = num_regions % batch_size
        if last_batch_size > 0:
            last_batch_start = batch_size * num_batches
            batches_bounds += [(last_batch_start, last_batch_start+last_batch_size)]
            num_batches += 1

        embeddings = torch.zeros([num_regions, self.embedding_size])
        
        for bounds in self.progress_bar(batches_bounds):
            yxs_batch = yxs[bounds[0]:bounds[1]]

            regions = self.slide_manager.get_multiple_regions(
                yxs_batch,
                n_cpus=self.n_cpus
            )
            
            regions = torch.from_numpy(regions).float().to(self.device)

            with torch.no_grad():
                embeddings_batch = self.fine_model(regions)
                
            embeddings[bounds[0]:bounds[1]] = embeddings_batch
            
#             output[:,yxs_batch[:,0], yxs_batch[:,1]] = embeddings.T
#             for yx, emb in zip(yxs_batch, embeddings):
#                 output[:,yx[0],yx[1]] = emb
                
        return yxs, embeddings
    
    def coarse_lists_to_map(self, yxs, embs, bg_emb=None):
                
        output = torch.zeros(
            (self.embedding_size,) + self.slide_manager.grid_size_yx,
            requires_grad=False
        )
        output[:,yxs[:,0], yxs[:,1]] = embs.T
        
        if not bg_emb is None:
            bg_yxs = np.where(self.slide_manager.foreground_map==False)
            output[:,bg_yxs[0],bg_yxs[1]] = bg_emb
            
        return output
    
    def get_embedding_size(self):
        in_imgs = torch.rand((2, 3, 1,1))
        temp_model = self.fine_model.to(torch.device("cpu"))
        out = temp_model(in_imgs)
        return out.size(1)
        
        
