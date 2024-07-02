import torch
from torch import nn



 

class ActivatorGatingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim,dim)
        self.proj_2 =  nn.Linear(dim,dim)
        self.proj_3 = nn.Linear(dim,dim)     
        self.gelu = nn.GELU()
       
             	   
    def forward(self, x):
        u, v = x, x 
        u = self.proj_1(u)
        u = self.gelu(u)
        
        
        v = self.proj_2(v)
        
       
        g = u * v
        
        out = self.proj_3(g)
        return out



class ActivatorBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.actgu = ActivatorGatingUnit(d_model)
      
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.actgu(x)           
        x = x + residual            
        out = x
        return out



class ACTIVATOR(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[ActivatorBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








