from einops import rearrange
import torch.nn as nn
import torch


class Depthwise3dCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Depthwise3dCNN, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=in_channels) # kernel_size=(3, 1, 1) -> 3 Frames
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class STAdapter(nn.Module):
  
    def __init__(self, in_channels, down_channels):  # 768, 384
        super(STAdapter, self).__init__()

        self.downscale = nn.Linear(in_channels, down_channels)
        self.upscale = nn.Linear(down_channels, in_channels)
        self.dw3dcnn = Depthwise3dCNN(down_channels, down_channels)
        self.act_fun = nn.GELU()
    
    def forward(self, x):
        # input shape -> [T, N, in_channels]
        x = self.downscale(x) # [T, N, down_channels]
        x = rearrange(x, 't n d -> d t n').unsqueeze(3) # [down_channels, T, N, 1]
        x = self.dw3dcnn(x) # [down_channels, T, N, 1]
        x = self.act_fun(rearrange(x.squeeze(), 'd t n -> t n d')) # [T, N, down_channels]
        x = self.upscale(x) # [T, N, in_channels]
        
        return x
    
      
class VitModel(nn.Module):
    def __init__(self, ~~~):
        super().__init__()
        ~~~~
        self.block = nn.ModuleList(~~)

        self.st_dapter = STAdapter(768, 384) # self.st_dapter = nn.ModuleList([ STAdapter(768, 384) for i in range(self.depth)])

        ~~~
    def forward(self, x):
        ~~~
        ''' 
        # ver1
        x = self.st_dapter(x)
        x = self.block(x)
        '''
        # ver2
        for layer_idx in range(self.depth):
            x = self.st_dapter[layer_idx](x)
            x = self.block[layer_idx](x)
        
        ~~~
        
        return x
          
        
    
    
    
    
