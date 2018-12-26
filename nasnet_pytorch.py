"""
    From https://github.com/ErikGartner/wasp-cifar10/blob/master/nasnet.py
    But I use pytorch here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    对x进行reshape，使其最终filter个数为num_of_filters，
    假如stride=2，则将长宽变成原来的一半；stride=1的话则长宽不变
"""
def factorized_reduction(x, num_of_filters, stride):
    in_channels = x.shape[1]

    if stride == 1:
        conv = nn.Conv2d(in_channels, num_of_filters, kernel_size=1, stride=1)
        bn = nn.BatchNorm2d(num_of_filters)
        return bn(conv(x))
        
    # stride = 2
    # 以一种特殊的方式使输出的filters个数为num_of_filters
    relu = nn.ReLU(inplace=False)
    conv1 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
    conv2 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
    bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
    pad = nn.ConstantPad2d((0,1,0,1),0)
    
    x = relu(x)
    y = pad(x)
    out = torch.cat([conv1(x), conv2(y[:,:,1:,1:])], dim=1)
    out = bn(out)
    
    return out
    

"""
   使x_1的长宽与x一致， num_of_filters是x的filter个数
"""
def reduce_prev_layer(x, x_1, num_of_filters):
    if x_1 is None:
        x_1 = x
        
    # x_1的长宽与x不同 （filters也可能不同）
    if x.shape[2] != x_1.shape[2]:
        x_1 = F.relu(x_1)
        x_1 = factorized_reduction(x_1, num_of_filters, stride=2)
        
    # 仅filters不同
    elif x.shape[1] != x_1.shape[1]:
        relu = nn.ReLU()
        conv = nn.Conv2d(x_1.shape[1], num_of_filters, 1, stride=1)
        bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
        x_1 = bn(conv(relu(x_1)))
        
    return x_1


"""
    使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
"""    
def create_cell_base(x, x_1, num_of_filters):
    x_1 = reduce_prev_layer(x, x_1, num_of_filters) # 使x_1的长宽与x一样
    
    relu = nn.ReLU()
    conv = nn.Conv2d(x.shape[1], num_of_filters, 1, stride=1)
    bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
    
    return bn(conv(relu(x))), x_1
    
"""
    综合考虑epoch和cell_idx，以确定drop probability
"""
def calc_drop_keep_prob(keep_prob, cell_idx, total_cells, epoch, max_epochs):
    if keep_prob == 1:
        return 1
        
    prob = keep_prob
    layer_ratio = (cell_idx + 1) / total_cells
    prob = 1 - layer_ratio * (1 - prob)
    current_ratio = epoch / max_epochs
    prob = (1 - current_ratio * (1 - prob))
    
    return prob
    
    
"""
    cell 中的 Separable conv 的定义
    stride=1时不改变长宽
    stride=2时长宽变成原来的一半
    使输出tensor的filter个数等于num_of_filters
"""
def sepConv(x, num_of_filters, kernel_size, stride=1, keep_prob=1):
    C_in = x.shape[1]
    C_out = num_of_filters
    
    # kernel_size = 3,5,7 => padding = 1,2,3
    padding = int((kernel_size - 1) / 2)
    
    op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0),
        nn.BatchNorm2d(C_out, eps=1e-3),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out),
        nn.Conv2d(C_out, C_out, kernel_size=1, padding=0),
        nn.BatchNorm2d(C_out, eps=1e-3),
        nn.Dropout2d(keep_prob)
    )
    
    return op(x)
    
 
"""
    cell 中的identity 操作
"""
def identity(x, num_of_filters):
    C_in = x.shape[1]
    C_out = num_of_filters
    
    op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(C_out, eps=1e-3),
    )
    
    if C_in != C_out:
        return op(x)
        
    return x
    
    
"""
    cell 中的 avg_layer 的定义
    stride=1时不改变长宽
    stride=2时长宽变成原来的一半
    使输出tensor的filter个数等于num_of_filters
"""
def avg_layer(x, num_of_filters, stride=1, keep_prob=1):
    C_in = x.shape[1]
    C_out = num_of_filters
    
    avg_pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    op = nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0),      
        nn.BatchNorm2d(C_out, eps=1e-3)
    )

    dropout = nn.Dropout2d(keep_prob)
    
    if C_in != C_out:
        return dropout(op(x))
        
    return dropout(avg_pool(x))

    

"""
    cell 中的 max_layer 的定义
    stride=1时不改变长宽
    stride=2时长宽变成原来的一半
    使输出tensor的filter个数等于num_of_filters
"""
def max_layer(x, num_of_filters, stride=1, keep_prob=1):
    C_in = x.shape[1]
    C_out = num_of_filters
    
    max_pool = nn.MaxPool2d(3, stride=stride, padding=1)
    
    op = nn.Sequential(
        nn.MaxPool2d(3, stride=stride, padding=1),
        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0),
        nn.BatchNorm2d(C_out, eps=1e-3)
    )
    
    dropout = nn.Dropout2d(keep_prob)
    
    if C_in != C_out:
        return dropout(op(x))
        
    return dropout(max_pool(x))
    
 
class Normal_cell(nn.Module):
    def __init__(self, filters, weight_decay, keep_prob, 
                 cell_idx, total_cells, max_epochs):
        super(Normal_cell, self).__init__()
        # set parameters
        self.filters = filters
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs
        
        print("Build normal_cell %d, filters=%d" %(cell_idx, filters))
        
        
    def forward(self, x, x_1, epoch):
        """
            先使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
        """
        x_1 = reduce_prev_layer(x, x_1, self.filters)
        
        cell_base_relu = nn.ReLU()
        cell_base_conv = nn.Conv2d(x.shape[1], self.filters, 1, stride=1)
        cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)
        
        x = cell_base_bn(cell_base_conv(cell_base_relu(x)))
        
        """
            再确定drop probability，综合考虑epoch和cell_idx
        """
        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)
                                      
        y1_a = sepConv(x, self.filters, 3, keep_prob=dp_prob)
        y1_b = identity(x, self.filters)
        y1 = y1_a + y1_b
        
        y2_a = sepConv(x_1, self.filters, 3, keep_prob=dp_prob)
        y2_b = sepConv(x, self.filters, 5, keep_prob=dp_prob)
        y2 = y2_a + y2_b
        
        y3_a = avg_layer(x, self.filters, keep_prob=dp_prob)
        y3_b = identity(x_1, self.filters)
        y3 = y3_a + y3_b
        
        y4_a = avg_layer(x_1, self.filters, keep_prob=dp_prob)
        y4_b = avg_layer(x_1, self.filters, keep_prob=dp_prob)
        y4 = y4_a + y4_b
        
        y5_a = sepConv(x_1, self.filters, 5, keep_prob=dp_prob)
        y5_b = sepConv(x_1, self.filters, 3, keep_prob=dp_prob)
        y5 = y5_a + y5_b
        
        return torch.cat([y1, y2, y3, y4, y5], dim=1)
        
        
        
class Reduction_cell(nn.Module):
    def __init__(self, filters, weight_decay, stride,
                 keep_prob, cell_idx, total_cells, max_epochs):
        super(Reduction_cell, self).__init__()
        # set parameters
        self.filters = filters
        self.stride = stride
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs
        
        print("Build reduction_cell %d, filters=%d" %(cell_idx, filters))
        
    def forward(self, x, x_1, epoch):
        """
            先使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
        """
        x_1 = reduce_prev_layer(x, x_1, self.filters)
        
        cell_base_relu = nn.ReLU()
        cell_base_conv = nn.Conv2d(x.shape[1], self.filters, 1, stride=1)
        cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)
        
        x = cell_base_bn(cell_base_conv(cell_base_relu(x)))
        
        """
            再确定drop probability，综合考虑epoch和cell_idx
        """
        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)
        
        y1_a = sepConv(x_1, self.filters, 7, stride=self.stride, keep_prob=dp_prob)
        y1_b = sepConv(x, self.filters, 5, stride=self.stride, keep_prob=dp_prob)
        y1 = y1_a + y1_b
        
        y2_a = max_layer(x, self.filters, stride=self.stride, keep_prob=dp_prob)
        y2_b = sepConv(x_1, self.filters, 7, stride=self.stride, keep_prob=dp_prob)
        y2 = y2_a + y2_b
        
        y3_a = avg_layer(x, self.filters, stride=self.stride, keep_prob=dp_prob)
        y3_b = sepConv(x_1, self.filters, 5, stride=self.stride, keep_prob=dp_prob)
        y3 = y3_a + y3_b
        
        z1_a = max_layer(x, self.filters, stride=self.stride, keep_prob=dp_prob)
        z1_b = sepConv(y1, self.filters, 3, keep_prob=dp_prob)
        z1 = z1_a + z1_b
        
        z2_a = avg_layer(y1, self.filters, keep_prob=dp_prob)
        z2_b = identity(y2, self.filters)
        z2 = z2_a + z2_b
        
        return torch.cat([z1, z2, y3], dim=1)
        
        
"""
   计算辅助的loss以帮助训练
"""        
def Auxhead(x, num_classes, final_filters):
    C_in = x.shape[1]
    
    relu = nn.ReLU()
    avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
    conv1 = nn.Conv2d(C_in, 128, kernel_size=1, stride=1)
    bn = nn.BatchNorm2d(128, eps=1e-3)
    
    x = relu(x)
    x = avg_pool(x)
    x = conv1(x)
    x = bn(x)
    x = relu(x)
    conv2 = nn.Conv2d(128, final_filters, kernel_size=x.shape[2], stride=1)
    x = conv2(x)
    x = relu(x)
    x = x.view(x.size()[0], -1)  # flatten the tensor
    fc = nn.Linear(final_filters, num_classes)
    
    x = fc(x)
    
    return nn.Softmax(dim=1)(x)
        
        
        
def Head(x, num_classes):
    relu = nn.ReLU()
    global_pooling = nn.AdaptiveAvgPool2d(1)
    
    x = relu(x)
    x = global_pooling(x)
    x = x.view(x.size()[0], -1)  # flatten the tensor
    fc = nn.Linear(x.shape[1], num_classes)
    x = fc(x)
    
    return nn.Softmax(dim=1)(x)

    

class NASnet(nn.Module):
    def __init__(self, num_normal_cells=6, num_blocks=2, weight_decay=1e-4,
                 num_classes=10, num_filters=32, stem_multiplier=3, filter_multiplier=2,
                 dimension_reduction=2, final_filters=768, 
                 dropout_prob=0.0, drop_path_keep=0.6, max_epochs=300):
                 
        super(NASnet, self).__init__()
        
        # set parameters
        self.num_normal_cells = num_normal_cells
        self.num_classes = num_classes
        self.final_filters = final_filters
        
        # set layers
        filters = num_filters
        self.stem = self.create_stem(filters, stem_multiplier)
        
        self.layer_norm1_1 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         1, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_2 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         2, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm1_3 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         3, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_4 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         4, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm1_5 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         5, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_6 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         6, num_normal_cells * num_blocks, max_epochs) 
                                                     
        filters *= filter_multiplier
        
        self.layer_redu1 = Reduction_cell(filters, weight_decay, dimension_reduction,
                                          drop_path_keep, 7, num_normal_cells * num_blocks, max_epochs)
          
        self.layer_norm2_1 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         8, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_2 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         9, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm2_3 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         10, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_4 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         11, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm2_5 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         12, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_6 = Normal_cell(filters, weight_decay, drop_path_keep,
                                         13, num_normal_cells * num_blocks, max_epochs) 
                                                     
        filters *= filter_multiplier
        
        self.layer_redu2 = Reduction_cell(filters, weight_decay, dimension_reduction,
                                          drop_path_keep, 14, num_normal_cells * num_blocks, max_epochs)
    
    
    def create_stem(self, filters, stem_multiplier):
        stem = nn.Sequential(
            nn.Conv2d(3, filters * stem_multiplier, kernel_size=3, stride=1, padding=1), # padding=SAME
            nn.BatchNorm2d(filters * stem_multiplier)
        )
        return stem
        
        
    def forward(self, input, epoch):
        x = self.stem(input)
        x_1 = None
        
        y = self.layer_norm1_1(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm1_2(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm1_3(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm1_4(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm1_5(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm1_6(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_redu1(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_1(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_2(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_3(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_4(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_5(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm2_6(x, x_1, epoch)
        x_1 = x
        x = y

        aux_head = Auxhead(x, self.num_classes, self.final_filters)
        
        y = self.layer_redu2(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = Head(x, self.num_classes)
        
        return y, aux_head
        

"""
    测试各函数的功能
"""
if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    y = factorized_reduction(x, 32, 2)
    print(y.shape)
    
    
    x = torch.randn(128, 32, 32, 32)
    x_1 = torch.randn(128, 16, 64, 64)
    x_1 = reduce_prev_layer(x, x_1, 32)
    print(x_1.shape)
    
    
    x = torch.randn(128, 32, 32, 32)
    y = torch.randn(128, 16, 64, 64)
    x, y = create_cell_base(x, y, 64)
    print(x.shape)
    print(y.shape)
    
    
    x = torch.randn(1, 32, 64, 64)
    y = avg_layer(x, num_of_filters=64, stride=1, keep_prob=1)
    z = avg_layer(x, num_of_filters=32, stride=2, keep_prob=1)
    print(y.shape)
    print(z.shape)
    
    x = torch.randn(1, 32, 64, 64)
    y = sepConv(x, num_of_filters=64, kernel_size=7, stride=2, keep_prob=1)
    print(y.shape)
    
    
    normal_cell = Normal_cell(64, weight_decay=1, keep_prob=1, 
                              cell_idx=1, total_cells=12, max_epochs=300)
    x = torch.randn(1, 64, 16, 16)
    x_1 = torch.randn(1, 32, 32, 32)
    y = normal_cell.forward(x, x_1, 1)
    print(y.shape)
                              
    
    reduction_cell = Reduction_cell(32, weight_decay=1, stride=2, keep_prob=1, 
                                    cell_idx=1, total_cells=12, max_epochs=300)
    x = torch.randn(1, 16, 32, 32)
    x_1 = torch.randn(1, 16, 32, 32)
    y = reduction_cell.forward(x, x_1, 1)
    print(y.shape)
    
    
    x = torch.randn(1, 128, 32, 32)
    y = Auxhead(x, num_classes=10, final_filters=768)
    print(y.shape)
    
    
    x = torch.randn(1, 128, 32, 32)
    y = Head(x, num_classes=10)
    print(y.shape)
    
    
    print("---------network----------")
    net = NASnet()
    
    x = torch.randn(1, 3, 32, 32)
    y, aux_head = net(x, 1)
    print(y.shape)
    print(aux_head.shape)