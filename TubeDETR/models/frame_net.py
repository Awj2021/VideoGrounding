"""
Frame net.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import ipdb

# VGG16
class VGG16(nn.Module):
    def __init__(self, train_from_scratch=False, fine_tune=False, num_blocks = 2, num_layers = 3):
        super(VGG16, self).__init__()

        if train_from_scratch:
            original_model = torchvision.models.vgg16(pretrained=False)

        else:
            original_model = torchvision.models.vgg16(pretrained=True)
            if not fine_tune:
                for param in original_model.parameters():
                    param.requires_grad = False
        # ipdb.set_trace()
        layers = list(original_model.children())[0][0:29]
        
        self.feat_extractor = nn.Sequential(*layers)
        
        dynamic_layers = []
        for _ in range(num_layers):
            dynamic_layers.append(self.build_layer(num_blocks))
        self.dynamic_layers = nn.Sequential(*dynamic_layers)

        
    def build_block(self):
        
        input_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)), 
            nn.ReLU()
        )
        
        score_layer  = nn.Sequential(
            nn.Linear(512, 1)
        )
        
        
        block = nn.ModuleDict({
            'input_layer': input_layer,
            'score_layer': score_layer    
        })
        return block
    
    def build_layer(self, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(self.build_block())
        
        blocks = nn.ModuleList(blocks)

        
        layer = nn.ModuleDict({
            'blocks': blocks
        })
        return layer
        
        
    def forward_layer(self, layer, input):
        batch_size = input.shape[0]
        scores = []
        input_xs = []
        
        for block in layer['blocks']:
            input_x = block['input_layer'](input)
            out = F.adaptive_max_pool2d(input_x, 1)
            out = out.squeeze(-1).squeeze(-1)
            # ipdb.set_trace()
            score = block['score_layer'](out)
            input_xs.append(input_x)
            scores.append(score)
        scores =torch.stack(scores, dim=1)
        discrete_scores = F.gumbel_softmax(scores, dim=1, hard=True)
        # ipdb.set_trace()
        discrete_scores = discrete_scores.reshape(list(discrete_scores.shape) + [1, 1])
        
        input_xs = torch.stack(input_xs, dim=1)
        # ipdb.set_trace()
        inputs = discrete_scores * input_xs
        inputs = inputs.sum(1)
        
        return inputs
        
             
        
    def forward(self, x):
        """
        Output: B x 512 x 14 x 14, for input of size B x 3 x 224 x 224
        Output: B x 512 x 20 x 20, for input of size B x 3 x 320 x 320
        """
        out = self.feat_extractor(x)
        
        residual = out
        for layer in self.dynamic_layers:
            out = self.forward_layer(layer, out)
            
        out = residual + out
        
        return out
        # return self.feat_extractor(x)

if __name__ == '__main__':
    temp = VGG16(False, True)
    ipdb.set_trace()