import torch
import torch.nn as nn
import ipdb

class WTransform2d(nn.Module):
    def __init__(self, batch_size, num_features, momentum=0.1, track_running_stats=True, eps=1e-3):
        super(WTransform2d, self).__init__()
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.batch_size = batch_size
        self.num_features = num_features
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros([self.num_features], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
            self.register_buffer('running_variance', torch.ones([self.num_features, self.num_features], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))

    def forward(self, x):
        mean = x.mean(0)
        if not self.training and self.track_running_stats: # for inference
        	mean = self.running_mean
        y = x - mean
        
        cov = torch.mm(y.t(), y) / y.size(0)
        if not self.training and self.track_running_stats: # for inference
            cov = self.running_variance
        cov_shrinked = (1 - self.eps) * cov + self.eps * torch.eye(self.num_features).cuda(0)
        
        inv_sqrt = torch.inverse(torch.cholesky(cov_shrinked))
        out = torch.mm(inv_sqrt, y.t()).t()
        
        if self.training and self.track_running_stats:
            self.running_mean = torch.add(self.momentum * mean.detach(), (1 - self.momentum) * self.running_mean)
            self.running_variance = torch.add(self.momentum * cov.detach(), (1 - self.momentum) * self.running_variance)
        	
        return out
        
        
class whitening_scale_shift(nn.Module):
    def __init__(self, batch_size, num_features, track_running_stats=True, affine=True):
        super(whitening_scale_shift, self).__init__()
        self.batch_size = batch_size
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.affine = affine
        
        self.wh = WTransform2d(self.batch_size, 
        								 self.num_features, 
        								 track_running_stats=self.track_running_stats)
        if self.affine:
        		self.gamma = nn.Parameter(torch.ones(self.num_features))
        		self.beta = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x):
        out = self.wh(x)
        if self.affine:
        	out = out * self.gamma + self.beta
         
        return out
    
            