import transforms as T
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]




class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform_cj = T.RandomApply([T.My_ColorJitter(0.4,0.4,0.4,0.1)], p=0.8, param=True)
        self.transform1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0))
        ])
        self.transform1_after = T.Compose([        
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0))
        ])
        self.transform2_after = T.Compose([     
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform3 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0))
        ])
        self.transform3_after = T.Compose([     
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform4 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip()
        ])
        self.transform4_after = T.Compose([        
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1_1_, x1_2_ = self.transform1(x), self.transform1(x)
        x1_1_c, color1_1 = self.transform_cj(x1_1_)
        x1_2_c, color1_2 = self.transform_cj(x1_2_)
        x1_1, x1_2 = self.transform1_after(x1_1_c), self.transform1_after(x1_2_c)

        x2_1_, x2_2_ = self.transform2(x), self.transform2(x)
        x2_1_c, color2_1 = self.transform_cj(x2_1_)
        x2_2_c, color2_2 = self.transform_cj(x2_2_)
        x2_1, x2_2 = self.transform2_after(x2_1_c), self.transform2_after(x2_2_c)

        x3_1_, x3_2_ = self.transform3(x), self.transform3(x)
        x3_1_c, color3_1 = self.transform_cj(x3_1_)
        x3_2_c, color3_2 = self.transform_cj(x3_2_)
        x3_1, x3_2 = self.transform3_after(x3_1_c), self.transform3_after(x3_2_c)

        x4_1_, x4_2_ = self.transform4(x), self.transform4(x)
        x4_1_c, color4_1 = self.transform_cj(x4_1_)
        x4_2_c, color4_2 = self.transform_cj(x4_2_)
        x4_1, x4_2 = self.transform4_after(x4_1_c), self.transform4_after(x4_2_c)

        return [x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, x4_1, x4_2, color2_1, color2_2, color3_1, color3_2, color4_1, color4_2]



def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)










