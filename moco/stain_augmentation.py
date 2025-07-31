import random
from typing import Tuple, Union

import torch
import torch.nn as nn


def _image_to_absorbance(image, min_val, max_val):
    min_val = torch.as_tensor(min_val, device=image.device, dtype=image.dtype)
    max_val = torch.as_tensor(max_val, device=image.device, dtype=image.dtype)
    image = torch.maximum(torch.minimum(image, max_val), min_val)
    absorbance = -torch.log(image / max_val)
    return absorbance


def image_to_absorbance(image, source_intensity=255.0, dtype=torch.float32):
    """Convert an image to units of absorbance (optical density).
    Parameters
    ----------
    image : ndarray
        The image to convert to absorbance. Can be single or multichannel.
    source_intensity : float, optional
        Reference intensity for `image`.
    dtype : numpy.dtype, optional
        The floating point precision at which to compute the absorbance.
    Returns
    -------
    absorbance : ndarray
        The absorbance computed from image.
    Notes
    -----
    If `image` has an integer dtype it will be clipped to range
    ``[1, source_intensity]``, while float image inputs are clipped to range
    ``[source_intensity/255, source_intensity]. The minimum is to avoid log(0).
    Absorbance is then given by
    .. math::
        absorbance = \\log{\\frac{image}{source_intensity}}.
    """

    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating point type")

    input_dtype = image.dtype
    image = image.to(dtype, copy=False)
    if source_intensity < 0:
        raise ValueError("Source transmitted light intensity must be a positive value.")
    source_intensity = float(source_intensity)
    if input_dtype.is_floating_point:
        min_val = source_intensity / 255.0
        max_val = source_intensity
    else:
        min_val = 1.0
        max_val = source_intensity

    absorbance = _image_to_absorbance(image, min_val, max_val)
    return absorbance


def _image_to_absorbance_matrix(
    image,
    source_intensity=240,
    image_type="intensity",
    channel_axis=0,
    dtype=torch.float32,
):
    """Convert image to an absorbance and reshape to (3, n_pixels).
    See ``image_to_absorbance`` for parameter descriptions
    """
    if (image_type == "intensity" and image.ndim == 4) or (
        image_type == "absorbance" and image.ndim == 3
    ):
        channel_axis += 1
    c = image.shape[channel_axis]
    if c != 3:
        raise ValueError("Expected an RGB image")

    if image_type == "intensity":
        absorbance = image_to_absorbance(
            image, source_intensity=source_intensity, dtype=dtype
        )
    elif image_type == "absorbance":
        absorbance = image.to(dtype, copy=True)
    else:
        raise ValueError("`image_type` must be either 'intensity' or 'absorbance'.")

    # reshape to form a (n_channels, n_pixels) matrix
    if (image_type == "intensity" and image.ndim == 3) or (
        image_type == "absorbance" and image.ndim == 2
    ):
        if channel_axis != 0:
            absorbance = torch.moveaxis(absorbance, source=channel_axis, destination=0)
        return absorbance.view(c, -1)
    elif (image_type == "intensity" and image.ndim == 4) or (
        image_type == "absorbance" and image.ndim == 3
    ):
        if channel_axis != 1:
            absorbance = torch.moveaxis(absorbance, source=channel_axis, destination=1)
        return absorbance.view(image.shape[0], c, -1)


def _absorbance_to_image_float(absorbance, source_intensity):
    return torch.exp(-absorbance) * source_intensity


def _absorbance_to_image_int(absorbance, source_intensity, min_val, max_val):
    rgb = torch.exp(-absorbance) * source_intensity
    # prevent overflow/underflow
    min_val = torch.as_tensor(min_val, device=absorbance.device, dtype=absorbance.dtype)
    max_val = torch.as_tensor(max_val, device=absorbance.device, dtype=absorbance.dtype)
    rgb = torch.minimum(torch.maximum(rgb, min_val), max_val)
    return torch.round(rgb)


def _absorbance_to_image_uint8(absorbance, source_intensity):
    rgb = torch.exp(-absorbance) * source_intensity
    # prevent overflow/underflow
    min_val = torch.as_tensor(0, device=absorbance.device, dtype=absorbance.dtype)
    max_val = torch.as_tensor(255, device=absorbance.device, dtype=absorbance.dtype)
    rgb = torch.minimum(torch.maximum(rgb, min_val), max_val)
    return torch.round(rgb).to(torch.uint8)


def absorbance_to_image(absorbance, source_intensity=255, dtype=torch.uint8):
    """Convert an absorbance (optical density) image back to a standard image.
    Parameters
    ----------
    absorbance : ndarray
        The absorbance image to convert back to a linear intensity range.
    source_intensity : float, optional
        Reference intensity for `image`. This should match what was used with
        ``rgb_to_absorbance`` when creating `absorbance`.
    dtype : numpy.dtype, optional
        The datatype to cast the output image to.
    Returns
    -------
    image : ndarray
        An image computed from the absorbance
    """
    # absorbance must be floating point
    absorbance_dtype = torch.promote_types(absorbance.dtype, torch.float16)
    absorbance = absorbance.to(absorbance_dtype, copy=False)

    if source_intensity < 0:
        raise ValueError("Source transmitted light intensity must be a positive value.")

    if dtype == torch.uint8:
        return _absorbance_to_image_uint8(absorbance, source_intensity)
    try:
        # round to nearest integer and cast to desired integer dtype
        iinfo = torch.iinfo(dtype)
        image = _absorbance_to_image_int(
            absorbance, source_intensity, iinfo.min, iinfo.max
        )
        return image.to(dtype, copy=False)
    except TypeError:
        return _absorbance_to_image_float(absorbance, source_intensity)


def _validate_image(image):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image must be of type torch.Tensor.")
    if image.min() < 0:
        raise ValueError("Image should not have negative values.")


def _prep_channel_axis(channel_axis, ndim):
    if (channel_axis < -ndim) or (channel_axis > ndim - 1):
        raise ValueError(f"`channel_axis={channel_axis}` exceeds image dimensions")
    return channel_axis % ndim


def _stain_extraction_pca(
    image,
    source_intensity=240,
    alpha=1,
    beta=0.345,
    *,
    channel_axis=0,
    image_type="intensity",
):
    """Extract the matrix of H & E stain coefficient from an image.
    Uses a method that selects stain vectors based on the angle distribution
    within a best-fit plane determined by principle component analysis (PCA)
    [1]_.
    Parameters
    ----------
    image : cp.ndarray
        RGB image to perform stain extraction on. Intensities should typically
        be within unsigned 8-bit integer intensity range ([0, 255]) when
        ``image_type == "intensity"``.
    source_intensity : float, optional
        Transmitted light intensity. The algorithm will clip image intensities
        above the specified `source_intensity` and then normalize by
        `source_intensity` so that `image` intensities are <= 1.0. Only used
        when `image_type=="intensity"`.
    alpha : float, optional
        Algorithm parameter controlling the ``[alpha, 100 - alpha]``
        percentile range used as a robust [min, max] estimate.
    beta : float, optional
        Absorbance (optical density) threshold below which to consider pixels
        as transparent. Transparent pixels are excluded from the estimation.
    Additional Parameters
    ---------------------
    channel_axis : int, optional
        The axis corresponding to color channels (default is the last axis).
    image_type : {"intensity", "absorbance"}, optional
        With the default `image_type` of `"intensity"`, the image will be
        transformed to `absorbance` units via ``image_to_absorbance``. If
        the input `image` is already an absorbance image, then `image_type`
        should be set to `"absorbance"` instead.
    Returns
    -------
    stain_coeff : cp.ndarray
        Stain attenuation coefficient matrix derived from the image, where
        the first column corresponds to H, the second column is E and the rows
        are RGB values.
    Notes
    -----
    The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
    difference is due to our use of the natural log instead of a decadic log
    (log10) when computing the absorbance.
    References
    ----------
    .. [1] M. Macenko et al., "A method for normalizing histology slides for
           quantitative analysis," 2009 IEEE International Symposium on
           Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
           doi: 10.1109/ISBI.2009.5193250.
           http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    _validate_image(image)
    channel_axis = _prep_channel_axis(channel_axis, image.ndim)
    if alpha < 0 or alpha > 100:
        raise ValueError("alpha must be a percentile in range [0, 100].")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    # convert to absorbance (optical density) matrix of shape (3, n_pixels)
    absorbance = _image_to_absorbance_matrix(
        image,
        source_intensity=source_intensity,
        image_type=image_type,
        channel_axis=channel_axis,
    )

    # remove transparent pixels
    mask = torch.all(absorbance > beta, dim=0)
    absorbance = absorbance[:, mask]
    if absorbance.numel() == 0:
        raise ValueError("All pixels of the input image are below the threshold.")

    # compute eigenvectors (do small 3x3 matrix calculations on the host)
    cov = torch.cov(absorbance)

    _, ev = torch.linalg.eigh(cov)
    ev = ev[:, [2, 1]]
    # flip to ensure positive first coordinate so arctan2 angles are about 0
    if ev[0, 0] < 0:
        ev[:, 0] *= -1
    if ev[0, 1] < 0:
        ev[:, 1] *= -1

    # project on the plane spanned by the eigenvectors
    projection = ev.T @ absorbance

    # find the vectors that span the whole data (min and max angles)
    phi = torch.atan2(projection[1], projection[0])
    min_phi, max_phi = torch.quantile(
        phi,
        torch.tensor((alpha, 100 - alpha), dtype=phi.dtype, device=phi.device) / 100,
    )

    # project back to absorbance space
    v_min = torch.stack([torch.cos(min_phi), torch.sin(min_phi)])
    v_max = torch.stack([torch.cos(max_phi), torch.sin(max_phi)])
    v1 = ev @ v_min
    v2 = ev @ v_max

    # Make Hematoxylin (H) first and eosin (E) second by comparing the
    # R channel value
    if v1[0] < v2[0]:
        v1, v2 = v2, v1
    stain_coeff = torch.stack((v1, v2), dim=-1)

    # renormalize columns to reduce numerical error
    stain_coeff /= torch.linalg.norm(stain_coeff, dim=0, keepdim=True)
    return stain_coeff


def stain_extraction_pca(
    image,
    source_intensity=240,
    alpha=1,
    beta=0.345,
    ref_stain_coeff=(
        (0.5626, 0.2159),
        (0.7201, 0.8012),
        (0.4062, 0.5581),
    ),
    channel_axis=0,
    image_type="intensity",
):
    """Extract the matrix of H & E stain coefficient from an image.
    Uses a method that selects stain vectors based on the angle distribution
    within a best-fit plane determined by principle component analysis (PCA)
    [1]_.
    Parameters
    ----------
    image : cp.ndarray
        RGB image to perform stain extraction on. Intensities should typically
        be within unsigned 8-bit integer intensity range ([0, 255]) when
        ``image_type == "intensity"``.
    source_intensity : float, optional
        Transmitted light intensity. The algorithm will clip image intensities
        above the specified `source_intensity` and then normalize by
        `source_intensity` so that `image` intensities are <= 1.0. Only used
        when `image_type=="intensity"`.
    alpha : float, optional
        Algorithm parameter controlling the ``[alpha, 100 - alpha]``
        percentile range used as a robust [min, max] estimate.
    beta : float, optional
        Absorbance (optical density) threshold below which to consider pixels
        as transparent. Transparent pixels are excluded from the estimation.
    Additional Parameters
    ---------------------
    channel_axis : int, optional
        The axis corresponding to color channels (default is the last axis).
    image_type : {"intensity", "absorbance"}, optional
        With the default `image_type` of `"intensity"`, the image will be
        transformed to `absorbance` units via ``image_to_absorbance``. If
        the input `image` is already an absorbance image, then `image_type`
        should be set to `"absorbance"` instead.
    Returns
    -------
    stain_coeff : cp.ndarray
        Stain attenuation coefficient matrix derived from the image, where
        the first column corresponds to H, the second column is E and the rows
        are RGB values.
    Notes
    -----
    The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
    difference is due to our use of the natural log instead of a decadic log
    (log10) when computing the absorbance.
    References
    ----------
    .. [1] M. Macenko et al., "A method for normalizing histology slides for
           quantitative analysis," 2009 IEEE International Symposium on
           Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
           doi: 10.1109/ISBI.2009.5193250.
           http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    if (image_type == "intensity" and image.ndim == 3) or (
        image_type == "absorbance" and image.ndim == 2
    ):
        return _stain_extraction_pca(
            image,
            source_intensity=source_intensity,
            alpha=alpha,
            beta=beta,
            channel_axis=channel_axis,
            image_type=image_type,
        )
    elif (image_type == "intensity" and image.ndim == 4) or (
        image_type == "absorbance" and image.ndim == 3
    ):
        stain_matrix = []
        for im in image:
            try:
                sm = _stain_extraction_pca(
                    im,
                    source_intensity=source_intensity,
                    alpha=alpha,
                    beta=beta,
                    channel_axis=channel_axis,
                    image_type=image_type,
                )
                if torch.det(sm.T @ sm) == 0:
                    raise ValueError
                stain_matrix.append(sm)
            except ValueError:
                stain_matrix.append(
                    torch.as_tensor(
                        ref_stain_coeff, dtype=image.dtype, device=image.device
                    )
                )
        stain_matrix = torch.stack(stain_matrix)
        return stain_matrix


def _get_raw_concentrations(src_stain_coeff, absorbance):
    # estimate the raw stain concentrations

    # pseudo-inverse
    coeff_pinv = torch.linalg.inv(
        src_stain_coeff.transpose(-1, -2) @ src_stain_coeff
    ) @ src_stain_coeff.transpose(-1, -2)
    if torch.any(torch.isnan(coeff_pinv)):
        # fall back to cp.linalg.lstsq if pseudo-inverse above failed
        return None
    else:
        conc_raw = coeff_pinv @ absorbance

    return conc_raw


def _normalized_from_concentrations(
    conc_raw,
    ref_stain_coeff,
    source_intensity,
    original_shape,
    channel_axis,
):
    """Determine normalized image from concentrations."""

    # reconstruct the image based on the reference stain matrix
    absorbance_norm = ref_stain_coeff @ conc_raw
    image_norm = absorbance_to_image(
        absorbance_norm, source_intensity=source_intensity, dtype=torch.uint8
    )

    # restore original shape for each channel
    if len(original_shape) == 4:
        n = original_shape[0]
        original_shape = original_shape[1:]
        squeeze = False
    else:
        n = 1
        squeeze = True
    spatial_shape = original_shape[:channel_axis] + original_shape[channel_axis + 1 :]
    image_norm = torch.reshape(image_norm, (n, 3) + spatial_shape)
    if squeeze:
        image_norm = image_norm.squeeze(0)

    # move channels from axis 0 to channel_axis
    if channel_axis != 0:
        image_norm = torch.moveaxis(image_norm, source=0, destination=channel_axis)
    # restore original shape
    return image_norm





"""
    Convert HSV to RGB.
    Input: hsv tensor of shape [..., 3] in range [0, 1]
    Output: rgb tensor of shape [..., 3] in range [0, 1]
"""
def hsv_to_rgb_torch(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = h * 6  # [0,6)
    i = torch.floor(h).long()
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    i = i % 6

    conditions = [
        (i == 0, torch.stack([v, t, p], dim=-1)),
        (i == 1, torch.stack([q, v, p], dim=-1)),
        (i == 2, torch.stack([p, v, t], dim=-1)),
        (i == 3, torch.stack([p, q, v], dim=-1)),
        (i == 4, torch.stack([t, p, v], dim=-1)),
        (i == 5, torch.stack([v, p, q], dim=-1)),
    ]

    rgb = torch.zeros_like(hsv)
    for cond, val in conditions:
        rgb = torch.where(cond.unsqueeze(-1), val, rgb)

    return rgb


"""
    Modified from Robin's version:
    The color picker is mine (using HSV for free stain)
"""
class StainAugmentor(nn.Module):
    def __init__(
        self,
        alpha_range: Union[float, Tuple[float, float]] = (0.5, 2),
        beta_range: Union[float, Tuple[float, float]] = (-0.15, 0.15),
        stain_matrix_mins: Tuple[float, float] = ((0, 0), (0.15, 0.2), (0.4, 0.6)),
        stain_matrix_maxs: Tuple[float, float] = ((1, 1), (0.6, 0.7), (0.8, 0.9)),
        p: float = 1, 
        color_system="hsv", 
        arg_gpu=None
    ):
        super().__init__()

        if isinstance(alpha_range, float):
            alpha_range = (1 - alpha_range, 1 + alpha_range)
        if isinstance(beta_range, float):
            beta_range = (-beta_range, beta_range)

        self.alpha_range = alpha_range
        self.beta_range = beta_range
        if arg_gpu is not None:
            self.stain_matrix_mins = torch.tensor(stain_matrix_mins).cuda(arg_gpu)
            self.stain_matrix_maxs = torch.tensor(stain_matrix_maxs).cuda(arg_gpu)
        else:
            self.stain_matrix_mins = torch.tensor(stain_matrix_mins).cuda()
            self.stain_matrix_maxs = torch.tensor(stain_matrix_maxs).cuda()
        self.p = p
        self.color_system = color_system

    def get_params(self, n, dtype=None, device=None):

        stains = (self.stain_matrix_maxs - self.stain_matrix_mins) * torch.rand(n, 3, 2, dtype=dtype, device=device) + self.stain_matrix_mins
        if self.color_system == "hsv":
            # Put the color channel at the end
            stain_perm = stains.permute(0, 2, 1)
            # Convert to rgb and permute back
            stains = hsv_to_rgb_torch(stain_perm).permute(0, 2, 1)
        stains = 1 - stains
        return {
            "alpha": (self.alpha_range[1] - self.alpha_range[0])
            * torch.rand(n, 2, dtype=dtype, device=device)
            + self.alpha_range[0],
            "beta": (self.beta_range[1] - self.beta_range[0])
            * torch.rand(n, 2, dtype=dtype, device=device)
            + self.beta_range[0],
            "stain_matrix": stains,
        }

    def forward(self, x):
        x *= 255
        params = self.get_params(x.shape[0], dtype=x.dtype, device=x.device)
        alpha = params["alpha"]
        beta = params["beta"]

        absorbance = _image_to_absorbance_matrix(x, channel_axis=0)
        stain_matrix = stain_extraction_pca(
            absorbance, image_type="absorbance", channel_axis=0
        )
        HE = _get_raw_concentrations(stain_matrix, absorbance)
        x /= 255
        if HE is None:
            return x
        stain_matrix = params["stain_matrix"]
        HE = torch.where(HE > 0.2, (HE * alpha[..., None] + beta[..., None]), HE)
        out = _normalized_from_concentrations(HE, stain_matrix, 240, x.shape, 0)
        for i in range(x.shape[0]):
            if random.random() > self.p:
                out[i] = x[i] 
        return out.to(x.dtype) / 255




def Robin_version(arg_gpu):
    stain_matrix_mins: Tuple[float, float] = ((0.4, 0.3), (0.5, 0.5), (0.3, 0.3)),
    stain_matrix_maxs: Tuple[float, float] = ((0.7, 0.5), (1, 1), (0.7, 0.7))
    return StainAugmentor(alpha_range=(0.67, 1.5), stain_matrix_mins=stain_matrix_mins,
                          stain_matrix_maxs=stain_matrix_maxs, color_system="rgb", arg_gpu=arg_gpu)


def realistic_version(arg_gpu):
    stain_matrix_mins: Tuple[float, float] = ((0.75, 0.80), (0.15, 0.2), (0.4, 0.6)),
    stain_matrix_maxs: Tuple[float, float] = ((0.80, 1), (0.6, 0.7), (0.8, 0.9))
    return StainAugmentor(alpha_range=(0.67, 1.5), stain_matrix_mins=stain_matrix_mins,
                          stain_matrix_maxs=stain_matrix_maxs, color_system="hsv", arg_gpu=arg_gpu)


def free_hue_version(arg_gpu):
    stain_matrix_mins: Tuple[float, float] = ((0, 0), (0.15, 0.2), (0.4, 0.6)),
    stain_matrix_maxs: Tuple[float, float] = ((1, 1), (0.6, 0.7), (0.8, 0.9))
    return StainAugmentor(alpha_range=(0.67, 1.5), stain_matrix_mins=stain_matrix_mins,
                          stain_matrix_maxs=stain_matrix_maxs, color_system="hsv", arg_gpu=arg_gpu)


def all_free_version(arg_gpu):
    stain_matrix_mins: Tuple[float, float] = ((0, 0), (0, 0), (0, 0))
    stain_matrix_maxs: Tuple[float, float] = ((1, 1), (1, 1), (1, 1))
    return StainAugmentor(alpha_range=(0.67, 1.5), stain_matrix_mins=stain_matrix_mins,
                          stain_matrix_maxs=stain_matrix_maxs, color_system="rgb", arg_gpu=arg_gpu)


    
