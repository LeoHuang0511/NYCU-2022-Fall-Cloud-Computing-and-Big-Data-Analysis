import torch
import torch.nn as nn
import numpy as np
#from exceptions.exceptions import InvalidBackboneError
import torchvision.models as models



class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, embedding_size=512):
        super(ResNetSimCLR, self).__init__()
        self.embedding_size = embedding_size
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=self.embedding_size),
                            "resnet50": models.resnet50(pretrained=False, num_classes=self.embedding_size),
                            #"resnet101": models.resnet101(pretrained=False, num_classes=self.embedding_size),
                            #"resnet152": models.resnet152(pretrained=False, num_classes=self.embedding_size)
                            }

        self.backbone = self._get_basemodel(base_model)
        self.dim_mlp = self.backbone.fc.in_features
        #print(self.dim_mlp)
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(in_features=self.dim_mlp, out_features=self.embedding_size)
        
        
        
        self.projection = nn.Sequential(
            nn.Linear(in_features=self.dim_mlp, out_features=self.dim_mlp),
            nn.BatchNorm1d(self.dim_mlp),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_mlp, out_features=self.embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        #self.newfc = nn.Linear(in_features=self.dim_mlp, out_features=self.embedding_size)


    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        
        return model

    def forward(self, x, return_embedding=False):
        #print(x.shape)
        embedding = self.backbone(x)
        #embedding = self.fc1(x)
        #x = self.fc1(x)
        #print(embedding.shape)
        
        if return_embedding:
           if self.dim_mlp != self.embedding_size:
               embedding = self.fc1(embedding)
           return embedding
        '''
        if return_embedding:
            
            return embedding
        '''
        return self.projection(embedding)


































'''
class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 7, 7),
                padding=(1, 3, 3),
                stride=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                stride=(1, 2, 2),
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        out_channels,
        temp_kernel=(1, 3, 3),
        spat_kernel=(1, 3, 3),
        temp_stride=(1, 1, 1),
        spat_stride=(1, 1, 1),
    ):
        super().__init__()
        temp_padding = tuple(k // 2 for k in temp_kernel)
        spat_padding = tuple(k // 2 for k in spat_kernel)
        self.blocks = nn.Sequential(
            nn.Conv3d(
                in_channels,
                inner_channels,
                kernel_size=temp_kernel,
                padding=temp_padding,
                stride=temp_stride,
                bias=False,
            ),
            nn.BatchNorm3d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inner_channels,
                out_channels,
                kernel_size=spat_kernel,
                padding=spat_padding,
                stride=spat_stride,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        out_channels,
        temp_kernel=(1, 1, 1),
        spat_kernel=(1, 3, 3),
        temp_stride=(1, 1, 1),
        spat_stride=(1, 1, 1),
    ):
        super().__init__()
        temp_padding = tuple(k // 2 for k in temp_kernel)
        spat_padding = tuple(k // 2 for k in spat_kernel)
        self.blocks = nn.Sequential(
            nn.Conv3d(
                in_channels,
                inner_channels,
                kernel_size=temp_kernel,
                padding=temp_padding,
                stride=temp_stride,
                bias=False,
            ),
            nn.BatchNorm3d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inner_channels,
                inner_channels,
                kernel_size=spat_kernel,
                padding=spat_padding,
                stride=spat_stride,
                bias=False,
            ),
            nn.BatchNorm3d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inner_channels,
                out_channels,
                kernel_size=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        ConvBlock,
        in_channels,
        inner_channels,
        out_channels,
        temp_kernel=(1, 1, 1),
        spat_kernel=(1, 1, 1),
        temp_stride=(1, 1, 1),
        spat_stride=(1, 1, 1),
    ):
        super().__init__()
        shortcut_stride = tuple(
            [x * y for x, y in zip(temp_stride, spat_stride)])
        if in_channels != out_channels or np.prod(shortcut_stride) != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1, 1),
                    stride=shortcut_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.main = ConvBlock(
            in_channels,
            inner_channels,
            out_channels,
            temp_kernel,
            spat_kernel,
            temp_stride,
            spat_stride
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.main(x) + self.shortcut(x)
        x = self.activation(x)
        return x


class ResStage(nn.Module):
    def __init__(
        self,
        ConvBlock,
        depth,
        in_channels,
        inner_channels,
        out_channels,
        temp_kernel=(1, 1, 1),
        spat_kernel=(1, 1, 1),
        temp_stride=(1, 1, 1),
        spat_stride=(1, 1, 1),
    ):
        super().__init__()
        self.blocks = []
        for idx in range(depth):
            resblock = ResBlock(
                ConvBlock,
                in_channels,
                inner_channels,
                out_channels,
                temp_kernel,
                spat_kernel,
                temp_stride,
                spat_stride,
            )
            self.blocks.append(resblock)

            in_channels = out_channels
            temp_stride = (1, 1, 1)
            spat_stride = (1, 1, 1)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class Head(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 5, 5),
        dropout=0.5
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.AvgPool3d(kernel_size, stride=1, padding=0),
            nn.Dropout(dropout),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        x = x.flatten(start_dim=1)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        ConvBlock,
        dim=64,
        depths=[3, 4, 6, 3],
        temp_kernels=[(1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)],
        spat_kernels=[(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)],
        temp_strides=[(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
        spat_strides=[(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        head_kernel=(4, 5, 5),
        head_dropout=0.5,
        num_classes=400,
    ):
        super().__init__()
        self.blocks = [Stem(3, dim)]
        in_channels = dim
        out_channels = in_channels * 4
        for idx in range(len(depths)):
            inner_channels = out_channels // 4
            stage = ResStage(
                ConvBlock,
                depths[idx],
                in_channels,
                inner_channels,
                out_channels,
                temp_kernels[idx],
                spat_kernels[idx],
                temp_strides[idx],
                spat_strides[idx],
            )
            self.blocks.append(stage)
            in_channels = out_channels
            out_channels = out_channels * 2
        self.blocks.append(
            Head(
                in_channels,
                num_classes,
                kernel_size=head_kernel,
                dropout=head_dropout
            )
        )
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class ResNet3d50(ResNet):
    def __init__(
        self,
        dim=64,
        head_kernel=(4, 5, 5),
        head_dropout=0.5,
        num_classes=50,
    ):
        super().__init__(
            BottleNeckBlock,
            dim=dim,
            depths=[3, 4, 6, 3],
            temp_kernels=[(1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)],
            spat_kernels=[(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)],
            temp_strides=[(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
            spat_strides=[(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
            head_kernel=head_kernel,
            head_dropout=head_dropout,
            num_classes=num_classes,
        )


class ResNet3d18(ResNet):
    def __init__(
        self,
        dim=64,
        head_kernel=(4, 5, 5),
        head_dropout=0.5,
        num_classes=50,
    ):
        super().__init__(
            BasicBlock,
            dim=dim,
            depths=[2, 2, 2, 2],
            temp_kernels=[(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)],
            spat_kernels=[(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)],
            temp_strides=[(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
            spat_strides=[(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
            head_kernel=head_kernel,
            head_dropout=head_dropout,
            num_classes=num_classes,
        )


if __name__ == '__main__':
    import torch
    resnet50 = ResNet3d50(num_classes=39)
    resnet18 = ResNet3d18(num_classes=39)
    x = torch.randn(8, 3, 8, 120, 120)
    y = resnet50(x)
    print(y.shape)
    y = resnet18(x)
    print(y.shape)
'''
