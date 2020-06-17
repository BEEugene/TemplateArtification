from albumentations import Compose, RandomBrightness, RandomGamma, Cutout, JpegCompression, \
    MultiplicativeNoise, RandomGridShuffle, Blur, Rotate, ElasticTransform, \
    VerticalFlip, HorizontalFlip, CenterCrop, OpticalDistortion, ImageCompression, CoarseDropout, \
    RandomBrightnessContrast, GridDistortion, OneOf, ChannelShuffle

class AlbumentationOptions:
    """
    class contains required augmentations pipelines
    """

    @staticmethod
    def box_segmentation_aug():
        return Compose([

            OneOf([
                RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
                RandomGamma(gamma_limit=50, p=0.5),
                ChannelShuffle(p=0.5)]),

            OneOf([
                ImageCompression(quality_lower=0, quality_upper=20, p=0.5),
                MultiplicativeNoise(multiplier=(0.3, 0.8), elementwise=True, per_channel=True, p=0.5),
                Blur(blur_limit=(15, 15), p=0.5)]),

            OneOf([
                CenterCrop(height=1000, width=1000, p=0.1),
                RandomGridShuffle(grid=(3, 3), p=0.2),
                CoarseDropout(max_holes=20, max_height=100, max_width=100, fill_value=53, p=0.2)]),

            OneOf([
                GridDistortion(p=0.5, num_steps=2, distort_limit=0.2),
                ElasticTransform(alpha=157, sigma=80, alpha_affine=196, p=0.5),
                OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5)]),

            OneOf([
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Rotate(limit=44, p=0.5)])
                   ])