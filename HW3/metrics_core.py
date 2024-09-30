

"""The core implementation of Inception Score and FID."""

from typing import List, Union, Tuple

import numpy as np
import torch
from scipy import linalg
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.hub import load_state_dict_from_url


#from .inception import InceptionV3
FID_WEIGHTS_URL = ('https://github.com/w86763777/pytorch-gan-metrics/releases/'
                   'download/v0.1.0/pt_inception-2015-12-05-6726825d.pth')
TORCH_VERSION = version.parse(torchvision.__version__)

class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation."""

    def __init__(self, in_channels, pool_features): # noqa
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):                           # noqa
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation."""

    def __init__(self, in_channels, channels_7x7):  # noqa
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):                           # noqa
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation."""

    def __init__(self, in_channels):    # noqa
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):               # noqa
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation."""

    def __init__(self, in_channels):    # noqa
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):               # noqa
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

def fid_inception_v3():
    """Build pretrained Inception model for FID computation.
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    if TORCH_VERSION < version.parse("0.13.0"):
        inception = models.inception_v3(
            pretrained=False,
            aux_logits=False,
            num_classes=1008,
            init_weights=False,
        )
    else:
        inception = models.inception_v3(
            weights=None,
            aux_logits=False,
            num_classes=1008,
            init_weights=False,
        )
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps."""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,      # First max pooling features
        192: 1,     # Second max pooling featurs
        768: 2,     # Pre-aux classifier features
        2048: 3,    # Final average pooling features
        1008: 4,    # softmax layer
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3.
        Args:
            output_blocks : List of int
                Indices of blocks to return features of. Possible values are:
                    - 0: corresponds to output of first max pooling
                    - 1: corresponds to output of second max pooling
                    - 2: corresponds to output which is fed to aux classifier
                    - 3: corresponds to output of final average pooling
                    - 4: corresponds to output of softmax
            resize_input : bool
                If true, bilinearly resizes input to width and height 299
                before feeding input to model. As the network without fully
                connected layers is fully convolutional, it should be able to
                handle inputs of arbitrary size, so resizing might not be
                strictly needed
            normalize_input : bool
                If true, scales the input from range (0, 1) to the range the
                pretrained Inception network expects, namely (-1, 1)
            requires_grad : bool
                If true, parameters of the model require gradients. Possibly
                useful for finetuning the network
            use_fid_inception : bool
                If true, uses the pretrained Inception model used in
                Tensorflow's FID implementation. If false, uses the pretrained
                Inception model available in torchvision. The FID Inception
                model has different weights and a slightly different structure
                from torchvision's Inception model. If you want to compute FID
                scores, you are strongly advised to set this parameter to true
                to get comparable results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = output_blocks
        self.last_needed_block = max(output_blocks)

        # assert self.last_needed_block <= 3, \
        #     'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            if TORCH_VERSION < version.parse("0.13.0"):
                inception = models.inception_v3(
                    pretrained=True,
                    init_weights=False)
            else:
                inception = models.inception_v3(
                    weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                )

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        if self.last_needed_block >= 4:
            inception.fc.bias = None
            self.blocks.append(inception.fc)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """Get Inception feature maps.
        Args:
            x : torch.FloatTensor,Input tensor of shape [B x 3 x H x W]. If
                `normalize_input` is True, values are expected to be in range
                [0, 1]; Otherwise, values are expected to be in range [-1, 1].
        Returns:
            List of torch.FloatTensor, corresponding to the selected output
            block, sorted ascending by index
        """
        outputs = [None for _ in range(len(self.output_blocks))]

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            if idx < 4:
                x = block(x)
            else:
                x = F.dropout(x, training=self.training)    # N x 2048 x 1 x 1
                x = torch.flatten(x, start_dim=1)           # N x 2048
                x = block(x)                                # N x 1000
                x = F.softmax(x, dim=1)

            if idx in self.output_blocks:
                order = self.output_blocks.index(idx)
                outputs[order] = x

            if idx == self.last_needed_block:
                break

        return outputs


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_inception_feature(
    images: Union[List[torch.FloatTensor], DataLoader],
    dims: List[int],
    batch_size: int = 50,
    use_torch: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device('cuda:0'),
) -> Union[torch.FloatTensor, np.ndarray]:
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to calculating
    features for FID and Inception Score.
    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        dims: List of int, see InceptionV3.BLOCK_INDEX_BY_DIM for
              available dimension.
        batch_size: int, The batch size for calculating activations. If
                    `images` is torch.utils.data.Dataloader, this argument is
                    ignored.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        verbose: Set verbose to False for disabling progress bar. Otherwise,
                 the progress bar is showing when calculating activations.
        device: the torch device which is used to calculate inception feature
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    assert all(dim in InceptionV3.BLOCK_INDEX_BY_DIM for dim in dims)

    is_dataloader = isinstance(images, DataLoader)
    if is_dataloader:
        num_images = min(len(images.dataset), images.batch_size * len(images))
        batch_size = images.batch_size
    else:
        num_images = len(images)

    block_idxs = [InceptionV3.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    model = InceptionV3(block_idxs).to(device)
    model.eval()

    if use_torch:
        features = [torch.empty((num_images, dim)).to(device) for dim in dims]
    else:
        features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        if is_dataloader:
            batch_images = next(looper)
        else:
            batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = batch_images.to(device)
        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims):
                if use_torch:
                    feature[start: end] = output.view(-1, dim)
                else:
                    feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features


def torch_cov(m, rowvar=False):
    """Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
           Each row of `m` represents a variable, and each column a single
           observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None): # noqa
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def calculate_frechet_inception_distance(
    acts: Union[torch.FloatTensor, np.ndarray],
    mu: np.ndarray,
    sigma: np.ndarray,
    use_torch: bool = False,
    eps: float = 1e-6,
    device: torch.device = torch.device('cuda:0'),
) -> float: # noqa
    if use_torch:
        m1 = torch.mean(acts, axis=0)
        s1 = torch_cov(acts, rowvar=False)
        mu = torch.tensor(mu).to(m1.dtype).to(device)
        sigma = torch.tensor(sigma).to(s1.dtype).to(device)
    else:
        m1 = np.mean(acts, axis=0)
        s1 = np.cov(acts, rowvar=False)
    return calculate_frechet_distance(m1, s1, mu, sigma, use_torch, eps)


def calculate_frechet_distance(
    mu1: Union[torch.FloatTensor, np.ndarray],
    sigma1: Union[torch.FloatTensor, np.ndarray],
    mu2: Union[torch.FloatTensor, np.ndarray],
    sigma2: Union[torch.FloatTensor, np.ndarray],
    use_torch: bool = False,
    eps: float = 1e-6,
) -> float:
    """Calculate Frechet Distance.
    Args:
        mu1: The sample mean over activations for a set of samples.
        sigma1: The covariance matrix over activations for a set of samples.
        mu2: The sample mean over activations for another set of samples.
        sigma2: The covariance matrix over activations for another set of
                samples.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        eps: prevent covmean from being singular matrix
    Returns:
        The Frechet Distance.
    """
    if use_torch:
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Run 50 itrs of newton-schulz to get the matrix sqrt of
        # sigma1 dot sigma2
        covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50)
        if torch.any(torch.isnan(covmean)):
            return float('nan')
        covmean = covmean.squeeze()
        out = (diff.dot(diff) +
               torch.trace(sigma1) +
               torch.trace(sigma2) -
               2 * torch.trace(covmean)).cpu().item()
    else:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = (diff.dot(diff) +
               np.trace(sigma1) +
               np.trace(sigma2) -
               2 * tr_covmean)
    return out


def calculate_inception_score(
    probs: Union[torch.FloatTensor, np.ndarray],
    splits: int = 10,
    use_torch: bool = False,
) -> Tuple[float, float]: # noqa
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        inception_score = torch.mean(scores).cpu().item()
        std = torch.std(scores).cpu().item()
    else:
        inception_score, std = (np.mean(scores), np.std(scores))
    del probs, scores
    return inception_score, std
