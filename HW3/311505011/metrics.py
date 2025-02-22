import os
from typing import List, Union, Tuple, Optional
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_tensor
from typing import List, Union, Tuple

import numpy as np
import torch
from scipy import linalg
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from metrics_core import (
    get_inception_feature,
    calculate_inception_score,
    calculate_frechet_inception_distance,
    torch_cov,
    InceptionV3)


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

def get_fid(
    images: Union[torch.FloatTensor, DataLoader],
    fid_stats_path: str,
    use_torch: bool = False,
    device: torch.device = torch.device('cuda:0'),
    **kwargs,
) -> float:
    """Calculate Frechet Inception Distance.
    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: Path to pre-calculated statistic.
        use_torch: When True, use torch to calculate FID. Otherwise, use numpy.
        **kwargs: The arguments passed to
                  `pytorch_gan_metrics.core.get_inception_feature`.
    Returns:
        FID
    """
    acts, = get_inception_feature(
        images, dims=[2048], use_torch=use_torch, device=device,**kwargs)

    # Frechet Inception Distance
    f = np.load(fid_stats_path)
    mu, sigma = f['mu'][:], f['sigma'][:]
    f.close()
    fid = calculate_frechet_inception_distance(acts, mu, sigma, use_torch)

    return fid