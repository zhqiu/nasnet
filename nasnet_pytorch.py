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
class Factorized_reduction(nn.Module):
    def __init__(self, in_channels, num_of_filters):
        super(Factorized_reduction, self).__init__()
        
        self.op_stride1 = nn.Sequential(
            nn.Conv2d(in_channels, num_of_filters, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_of_filters),
        )
        
        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ConstantPad2d((0,1,0,1),0)
        self.conv1 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
        self.conv2 = nn.Conv2d(in_channels, num_of_filters // 2, 1, stride=2)
        self.bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
        self.pad = nn.ConstantPad2d((0,1,0,1),0)
        
    def forward(self, x, stride):
        if stride == 1:
            return self.op_stride1(x)
            
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv1(x), self.conv2(y[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        
        return out
        

"""
   使x_1的长宽与x一致， num_of_filters是x的filter个数
   并使得x_1的filter个数为num_of_filters
"""
class Reduce_prev_layer(nn.Module):
    def __init__(self, x_width, x_1_width, x_1_channels, num_of_filters):
        super(Reduce_prev_layer, self).__init__()
        
        self.x_width = x_width
        self.x_1_width = x_1_width
        self.x_1_channels = x_1_channels
        self.num_of_filters = num_of_filters
        self.relu = nn.ReLU()
        self.factorized_reduction = Factorized_reduction(x_1_channels, num_of_filters)
        self.conv = nn.Conv2d(x_1_channels, num_of_filters, 1, stride=1)
        self.bn = nn.BatchNorm2d(num_of_filters, eps=1e-3)
        
        # 对第一个x_1进行处理
        self.conv_ = nn.Conv2d(96, num_of_filters, 1, stride=1)
        self.bn_ = nn.BatchNorm2d(num_of_filters, eps=1e-3)
        
    def forward(self, x, x_1):
        if x_1 is None:
            x_1 = x
            
        # x_1的长宽与x不同 （filters也可能不同）
        if self.x_width != self.x_1_width:
            x_1 = self.relu(x_1)
            x_1 = self.factorized_reduction(x_1, stride=2)
        
        # 仅filter个数不同
        elif self.x_1_channels != self.num_of_filters:
            x_1 = self.relu(x_1)
            x_1 = self.conv(x_1)
            x_1 = self.bn(x_1)
            
        return x_1
 
 
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
class SepConv(nn.Module):
    def __init__(self, C_in, num_of_filters, kernel_size, stride=1):
        super(SepConv, self).__init__()
        C_out = num_of_filters
        
        # kernel_size = 3,5,7 => padding = 1,2,3
        padding = int((kernel_size - 1) / 2)
        
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(C_out, eps=1e-3),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out),
            nn.Conv2d(C_out, C_out, kernel_size=1, padding=0),
            nn.BatchNorm2d(C_out, eps=1e-3),
        )
        
    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)
        return dropout(self.op(x))
        
    
 
"""
    cell 中的identity 操作
"""
class identity(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(identity, self).__init__()
        self.C_in = C_in
        self.C_out = num_of_filters
        
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(self.C_in, self.C_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C_out, eps=1e-3),
        )
        
    def forward(self, x):
        if self.C_in != self.C_out:
            return self.op(x)
            
        return x

    
    
"""
    cell 中的 avg_layer 的定义
    stride=1时不改变长宽
    stride=2时长宽变成原来的一半
    使输出tensor的filter个数等于num_of_filters
""" 
class avg_layer(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(avg_layer, self).__init__()
        
        self.C_in = C_in
        self.C_out = num_of_filters
        
        self.avg_pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.Conv2d(self.C_in, self.C_out, 1, stride=1, padding=0),      
            nn.BatchNorm2d(self.C_out, eps=1e-3)
        )
        
    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)
        
        if self.C_in != self.C_out:
            return dropout(self.op(x))
        
        return dropout(self.avg_pool(x))
   

"""
    cell 中的 max_layer 的定义
    stride=1时不改变长宽
    stride=2时长宽变成原来的一半
    使输出tensor的filter个数等于num_of_filters
    
    C_in: 输入tensor的filter个数
    num_of_filters：输出tensor的filters个数
"""
class max_layer(nn.Module):
    def __init__(self, C_in, num_of_filters, stride=1):
        super(max_layer, self).__init__()
        
        self.C_in = C_in
        self.C_out = num_of_filters
        
        self.max_pool = nn.MaxPool2d(3, stride=stride, padding=1)
        
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.Conv2d(self.C_in, self.C_out, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.C_out, eps=1e-3)
        )
        
    def forward(self, x, keep_prob=1):
        dropout = nn.Dropout2d(keep_prob)
        
        if self.C_in != self.C_out:
            return dropout(self.op(x))
            
        return dropout(self.max_pool(x))
        
    
 
class Normal_cell(nn.Module):
    """ 
        x_channels是x的filter的个数
        filters是想让x变成的filter的个数
    """
    def __init__(self, x_width, x_1_width, x_channels, x_1_channels,
                 filters, keep_prob, cell_idx, total_cells, max_epochs):
        super(Normal_cell, self).__init__()
        
        # set parameters
        self.filters = filters
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs
        
        # set base layer 
        # 使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
        self.cell_base_relu = nn.ReLU()
        self.cell_base_conv = nn.Conv2d(x_channels, self.filters, 1, stride=1)
        self.cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)
        
        self.reduce_prev_layer = Reduce_prev_layer(x_width, x_1_width, x_1_channels, filters)
        
        self.avg_layer_y3_a = avg_layer(filters, filters)
        self.avg_layer_y4_a = avg_layer(filters, filters)
        self.avg_layer_y4_b = avg_layer(filters, filters)
        self.identity_y1_b = identity(filters, filters)
        self.identity_y3_b = identity(filters, filters)
        self.sepConv_y1_a = SepConv(filters, filters, kernel_size=3)
        self.sepConv_y2_a = SepConv(filters, filters, kernel_size=3)
        self.sepConv_y2_b = SepConv(filters, filters, kernel_size=5)
        self.sepConv_y5_a = SepConv(filters, filters, kernel_size=5)
        self.sepConv_y5_b = SepConv(filters, filters, kernel_size=3)
        
        print("Build normal_cell %d, input filters=%d, output filters(one branch)=%d" 
              %(cell_idx, x_channels, filters))
        
        
    def forward(self, x, x_1, epoch):
        """
            先使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
        """
        x_1 = self.reduce_prev_layer(x, x_1)
        
        x = self.cell_base_bn(self.cell_base_conv(self.cell_base_relu(x)))

        
        """
            再确定drop probability，综合考虑epoch和cell_idx
        """
        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)
                                      
        y1_a = self.sepConv_y1_a(x, keep_prob=dp_prob)
        y1_b = self.identity_y1_b(x)
        y1 = y1_a + y1_b
               
        y2_a = self.sepConv_y2_a(x_1, keep_prob=dp_prob)
        y2_b = self.sepConv_y2_b(x, keep_prob=dp_prob)
        y2 = y2_a + y2_b
        
        y3_a = self.avg_layer_y3_a(x, keep_prob=dp_prob)
        y3_b = self.identity_y3_b(x_1)
        y3 = y3_a + y3_b
        
        y4_a = self.avg_layer_y4_a(x_1, keep_prob=dp_prob)
        y4_b = self.avg_layer_y4_b(x_1, keep_prob=dp_prob)
        y4 = y4_a + y4_b
        
        y5_a = self.sepConv_y5_a(x_1, keep_prob=dp_prob)
        y5_b = self.sepConv_y5_b(x_1, keep_prob=dp_prob)
        y5 = y5_a + y5_b
        
        return torch.cat([y1, y2, y3, y4, y5], dim=1)
        
        
        
class Reduction_cell(nn.Module):
    def __init__(self, x_width, x_1_width, x_channels, x_1_channels, 
                 filters, stride,
                 keep_prob, cell_idx, total_cells, max_epochs):
        super(Reduction_cell, self).__init__()
        # set parameters
        self.filters = filters
        self.stride = stride
        self.keep_prob = keep_prob
        self.cell_idx = cell_idx
        self.total_cells = total_cells
        self.max_epochs = max_epochs
        
        # set blocks
        self.cell_base_relu = nn.ReLU()
        self.cell_base_conv = nn.Conv2d(x_channels, self.filters, 1, stride=1)
        self.cell_base_bn = nn.BatchNorm2d(self.filters, eps=1e-3)
        
        self.reduce_prev_layer = Reduce_prev_layer(x_width, x_1_width, x_1_channels, filters)
        
        self.max_layer_y2_a = max_layer(filters, filters, stride)
        self.max_layer_z1_a = max_layer(filters, filters, stride)
        self.avg_layer_y3_a = avg_layer(filters, filters, stride)
        self.avg_layer_z2_a = avg_layer(filters, filters)          # stride=1
        self.identity_z2_b = identity(filters, filters)
        self.sepConv_y1_a = SepConv(filters, filters, 7, stride)
        self.sepConv_y1_b = SepConv(filters, filters, 5, stride)
        self.sepConv_y2_b = SepConv(filters, filters, 7, stride)
        self.sepConv_y3_b = SepConv(filters, filters, 5, stride)
        self.sepConv_z1_b = SepConv(filters, filters, 3)
        
        print("Build reduction_cell %d, input filters=%d, output filters(one branch)=%d" 
              %(cell_idx, x_channels, filters))
        
    def forward(self, x, x_1, epoch):
        """
            先使得x_1的长宽与x相同，并将x_1和x的filters个数变成num_of_filters
        """
        x_1 = self.reduce_prev_layer(x, x_1)
 
        x = self.cell_base_bn(self.cell_base_conv(self.cell_base_relu(x)))
        
        """
            再确定drop probability，综合考虑epoch和cell_idx
        """
        dp_prob = calc_drop_keep_prob(self.keep_prob, self.cell_idx,
                                      self.total_cells, epoch, self.max_epochs)
        
        y1_a = self.sepConv_y1_a(x_1, keep_prob=dp_prob)
        y1_b = self.sepConv_y1_b(x, keep_prob=dp_prob)
        y1 = y1_a + y1_b
        
        y2_a = self.max_layer_y2_a(x, keep_prob=dp_prob)
        y2_b = self.sepConv_y2_b(x_1, keep_prob=dp_prob)
        y2 = y2_a + y2_b
        
        y3_a = self.avg_layer_y3_a(x, keep_prob=dp_prob)
        y3_b = self.sepConv_y3_b(x_1, keep_prob=dp_prob)
        y3 = y3_a + y3_b
        
        z1_a = self.max_layer_z1_a(x, keep_prob=dp_prob)
        z1_b = self.sepConv_z1_b(y1, keep_prob=dp_prob)
        z1 = z1_a + z1_b
        
        z2_a = self.avg_layer_z2_a(y1, keep_prob=dp_prob)
        z2_b = self.identity_z2_b(y2)
        z2 = z2_a + z2_b
        
        return torch.cat([z1, z2, y3], dim=1)
        

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
        
"""
   计算辅助的loss以帮助训练
"""   
class Auxhead(nn.Module):
    def __init__(self, C_in, x_width, num_classes, final_filters):
        super(Auxhead, self).__init__()
        
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(C_in, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-3)
        self.conv2 = nn.Conv2d(128, final_filters, kernel_size=x_width, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(final_filters, num_classes)
        
    def forward(self, x):
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return nn.Softmax(dim=1)(x)
    

class Head(nn.Module):
    def __init__(self, x_channels, num_classes):
        super(Head, self).__init__()
        
        self.relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.fc = nn.Linear(x_channels, num_classes)
        
    def forward(self, x):
        x = self.relu(x)
        x = self.global_pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return nn.Softmax(dim=1)(x)
        
    

class NASnet(nn.Module):
    def __init__(self, num_normal_cells=6, num_blocks=3,
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
        
        # 对应norm1_1，x_1 = x
        self.layer_norm1_1 = Normal_cell(32, 32, filters * stem_multiplier, filters * stem_multiplier, 
                                         filters, drop_path_keep,
                                         1, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_2 = Normal_cell(32, 32, filters * 5, filters * stem_multiplier,
                                         filters, drop_path_keep,
                                         2, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm1_3 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         3, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_4 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         4, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm1_5 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         5, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm1_6 = Normal_cell(32, 32, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         6, num_normal_cells * num_blocks, max_epochs) 
        
        old_filters = filters
        filters *= filter_multiplier
        
        self.layer_redu1 = Reduction_cell(32, 32, old_filters * 5, old_filters * 5,
                                          filters, dimension_reduction,
                                          drop_path_keep, 7, num_normal_cells * num_blocks, max_epochs)
          
        self.layer_norm2_1 = Normal_cell(16, 32, filters * 3, old_filters * 5,
                                         filters, drop_path_keep,
                                         8, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_2 = Normal_cell(16, 16, filters * 5, filters * 3, 
                                         filters, drop_path_keep,
                                         9, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm2_3 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         10, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_4 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         11, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm2_5 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         12, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm2_6 = Normal_cell(16, 16, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         13, num_normal_cells * num_blocks, max_epochs) 
        
        old_filters = filters
        filters *= filter_multiplier
        
        self.layer_redu2 = Reduction_cell(16, 16, old_filters * 5, old_filters * 5,
                                          filters, dimension_reduction,
                                          drop_path_keep, 14, num_normal_cells * num_blocks, max_epochs)
                                          
        self.layer_norm3_1 = Normal_cell(8, 16, filters * 3, old_filters * 5,
                                         filters, drop_path_keep,
                                         15, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_2 = Normal_cell(8, 8, filters * 5, filters * 3, 
                                         filters, drop_path_keep,
                                         16, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm3_3 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         17, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_4 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         18, num_normal_cells * num_blocks, max_epochs) 
        self.layer_norm3_5 = Normal_cell(8, 8, filters * 5, filters * 5,
                                         filters, drop_path_keep,
                                         19, num_normal_cells * num_blocks, max_epochs)
        self.layer_norm3_6 = Normal_cell(8, 8, filters * 5, filters* 5, 
                                         filters, drop_path_keep,
                                         20, num_normal_cells * num_blocks, max_epochs)
                                         
        self.head = Head(640, num_classes)
        self.auxhead = Auxhead(320, 4, num_classes, final_filters)
    
    
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

        aux_head = self.auxhead(x)
        
        y = self.layer_redu2(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_1(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_2(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_3(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_4(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_5(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.layer_norm3_6(x, x_1, epoch)
        x_1 = x
        x = y
        
        y = self.head(x)
        
        return y, aux_head
        

"""
    测试各函数的功能
"""
if __name__ == "__main__":
    """
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
    
    
    normal_cell = Normal_cell(64, keep_prob=1, 
                              cell_idx=1, total_cells=12, max_epochs=300)
    x = torch.randn(1, 64, 16, 16)
    x_1 = torch.randn(1, 32, 32, 32)
    y = normal_cell.forward(x, x_1, 1)
    print(y.shape)
                              
    
    reduction_cell = Reduction_cell(32, stride=2, keep_prob=1, 
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
    """ 
    
    print("---------network----------")
    net = NASnet()
    
    x = torch.randn(1, 3, 32, 32)
    y, aux_head = net(x, 1)
    print(y.shape)
    print(aux_head.shape)