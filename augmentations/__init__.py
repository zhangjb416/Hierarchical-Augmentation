from .simsiam_aug import SimSiamTransform
from .simsiam_pred_aug import SimSiamTransform_pred
from .simsiam_pred_aug import SimSiamTransform_pred_eval
from .eval_aug import Transform_single
def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        elif name == 'simsiam_pred':
            augmentation = SimSiamTransform_pred(image_size)
        elif name == 'simsiam_pred_eval':
            augmentation = SimSiamTransform_pred_eval(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








