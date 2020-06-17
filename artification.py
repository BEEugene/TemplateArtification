import logging
import os
import random
import shutil

import cv2
from tqdm import tqdm

from augmentationoptions.albumentationsoptions import AlbumentationOptions
from utilities.filesystem import FolderIterator, PathSupport
from utilities.inputoutput import IO


class ImageArtification_self:
    """Resize the input to the given height and width.

        Args:
            p (float): probability of applying the transform. Default: 1.
            background_folder_path (str): folder where subfolders with unlabled images stored.
            foreground_folder_path (str): folder where image and labled mask subfolders stored.
            flag (OpenCV flag): flag that is used to specify the seamlessClone algorithm. Should be one of:
                cv2.MIXED_CLONE, cv2.NORMAL_CLONE.
                Default: cv2.INTER_LINEAR.

        Targets:
            image, mask, bboxes

        Image types:
            uint8, float32
        """

    def __init__(self, background_folder_path=None, foreground_folder_path=None,
                 flag=cv2.MIXED_CLONE, bg_param=None, fg_param=None, name_val_match=None,
                 always_apply=True, p_b=.7, p_f=1, p_f_freq=.7):
        self.logger = logging.getLogger("ImageArtification_self")
        self.background_folder = background_folder_path
        self.foreground_folder = foreground_folder_path
        self.proc_params(bg_param, fg_param)
        self.name_val_match = name_val_match
        self.masks = []
        self.all_masks = {}
        self.all_foregrounds = {}
        self.proc_folders()
        self.cnts = None
        self.coords = []
        self.artificial_background = None
        self.artificial_foreground = None
        self.foregrounds = None
        self.image = None
        self.arti_height = None
        self.arti_width = None
        self.im_height = None
        self.im_width = None
        self.coord_d = {}
        self.p_b = p_b
        self.p_f = p_f
        self.p_f_freq = p_f_freq

    def proc_params(self, bg_param, fg_param):
        if not bg_param:
            self.bg_param = {}
        if not fg_param:
            self.fg_param = {}

    def proc_folders(self):

        if self.background_folder:
            self.bg_folds = []
            for element in os.listdir(self.background_folder):
                if os.path.isdir(os.path.join(self.background_folder, element)):
                    self.bg_folds.append(element)

            for folder in self.bg_folds:
                if folder not in self.bg_param:
                    print("params for {fld} not found".format(fld=folder))

        if self.foreground_folder:
            self.fg_folds = os.listdir(self.foreground_folder)
            for folder in self.fg_folds:
                if folder not in self.fg_param:
                    print("params for {fld} not found".format(fld=folder))

    def apply(self, image, artificial, cnts, **param):
        return artificate(image, artificial, cnts)

    def apply_to_mask(self, image, **params):
        return artificate(image)

    def fit_background(self):
        pass

    def __fit_borders(self, width=None, height=None):
        # transform = albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True) # didn't work
        # new=transform.apply(self.artificial_image)
        if not isinstance(self.artificial_background, type(None)):
            if self.arti_height < self.im_height or self.arti_width < self.im_width:
                clac_dif_h = self.im_height - self.arti_height
                clac_dif_w = self.im_height - self.arti_width
                ext_height = (int(clac_dif_h / 2)) if clac_dif_h > 0 else 0
                top, bottom = ext_height + 10, ext_height + 10
                ext_width = int(clac_dif_w / 2) if clac_dif_w > 0 else 0
                left, right = ext_width + 10, ext_width + 10
                self.artificial_background = cv2.copyMakeBorder(self.artificial_background, top, bottom, left, right,
                                                                cv2.BORDER_REFLECT_101)[:self.im_height, :self.im_width,
                                             :]
            if self.arti_height > self.im_height or self.arti_width > self.im_width:
                self.artificial_background = cv2.resize(self.artificial_background,
                                                        (self.im_width,
                                                         self.im_height))  # to prevent background oversize
                self.artificial_background = self.artificial_background[:self.im_height, :self.im_width, :]
        if not isinstance(self.artificial_foreground, type(None)) and height and width:
            if self.arti_height < height or self.arti_width < width:
                clac_dif_h = height - self.arti_height
                clac_dif_w = height - self.arti_width
                ext_height = (int(clac_dif_h / 2)) if clac_dif_h > 0 else 0
                top, bottom = ext_height, ext_height
                ext_width = int(clac_dif_w / 2) if clac_dif_w > 0 else 0
                left, right = ext_width, ext_width
                if self.arti_height * 2 < height:
                    self.artificial_foreground = cv2.copyMakeBorder(self.artificial_foreground, top, bottom, left,
                                                                    right,
                                                                    cv2.BORDER_REFLECT_101)[:height, :width, :]
                self.artificial_foreground = cv2.resize(self.artificial_foreground, (width, height))
            if self.arti_height > height or self.arti_width > width:
                self.artificial_foreground = self.artificial_foreground[:height, :width, :]

    def __choose_object(self, chosen):
        logger = self.logger.getChild("__chosen_obgect")
        chosen_path = os.path.join(self.background_folder, chosen)
        chosen_object = random.choice(os.listdir(chosen_path))
        artificial_image_path = os.path.join(chosen_path, chosen_object)
        logger.debug(("artificial_image_path", artificial_image_path))
        logger.debug(("chosen_path", chosen_path))
        return artificial_image_path

    def get_params_dependent_on_targets(self, image, mask):

        logger = self.logger.getChild("get_params_dependent_on_targets")
        self.image = image
        logger.debug(("self.image.shape", self.image.shape))
        self.im_height, self.im_width = self.image.shape[:2]  # params['image'].shape[:2]
        self.cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(self.cnts))
        coords = []

        if self.foreground_folder:
            chosen_fg = random.choice(self.fg_folds)
        if self.background_folder:
            chosen_bg = random.choice(self.bg_folds)
        medians = {}
        for ind, item in enumerate(self.cnts):
            rect = cv2.boundingRect(item)
            coords.append(rect)
            self.coord_d[ind] = rect
            x, y, w, h = rect
            crop_mask = mask[y:y + h, x:x + w]
            median = int(np.median(crop_mask))
            medians[ind] = median
            logger.debug(("crop_mask.shape", crop_mask.shape))
            logger.debug(("crop_mask", crop_mask))
            mask_final = cv2.merge((crop_mask, crop_mask, crop_mask))
            # GUI_interaction.imgshow(mask_final)
            self.all_masks[ind] = mask_final
            self.all_foregrounds[ind] = None
            logger.debug(("mask_final.shape", mask_final.shape))
            logger.debug(("median", median))
            logger.debug(("median", median))
            logger.debug(("self.name_val_match[chosen_fg] ", self.name_val_match[chosen_fg]))

        if self.background_folder and random.random() < self.p_b:
            chosen_path = os.path.join(self.background_folder, chosen_bg)
            chosen_object = random.choice(os.listdir(chosen_path))
            artificial_image_path = os.path.join(chosen_path, chosen_object)
            logger.debug(("artificial_image_path", artificial_image_path))
            logger.debug(("chosen_path", chosen_path))
            self.artificial_background = IO.read(artificial_image_path)
            self.arti_height, self.arti_width = self.artificial_background.shape[:2]
            logger.debug(("self.arti_height < self.im_height or self.arti_width < self.im_width",
                          self.arti_height < self.im_height or self.arti_width < self.im_width))

            logger.debug(("self.artificial_image.shape", self.artificial_background.shape))
            self.__fit_borders()
            # for ind, item in self.coord_d:

            # print(self.artificial_image.shape)
            # todo: create artificial mask when transforming the artificial image
            a_height, a_width = self.artificial_background.shape[:2]  # params['image'].shape[:2]
            artificial_mask = None
        if self.foreground_folder and random.random() < self.p_f:
            self.foregrounds = []
            if self.name_val_match:
                names = list(self.name_val_match)
                values = list(self.name_val_match.values())
                if chosen_fg in names:
                    for ind, coord in self.coord_d.items():
                        if medians[ind] == self.name_val_match[chosen_fg]:
                            x, y, w, h = coord
                            if random.random() < self.p_f_freq:
                                chosen_path = os.path.join(self.foreground_folder, chosen_fg)
                                chosen_object = random.choice(os.listdir(chosen_path))
                                artificial_image_path = os.path.join(chosen_path, chosen_object)
                                logger.debug(("artificial_image_path", artificial_image_path))
                                logger.debug(("chosen_path", chosen_path))
                                self.artificial_foreground = IO.read(artificial_image_path)
                                self.arti_height, self.arti_width = self.artificial_foreground.shape[:2]
                                logger.debug(("self.arti_height < self.im_height or self.arti_width < self.im_width",
                                              self.arti_height < self.im_height or self.arti_width < self.im_width))
                                logger.debug(("self.artificial_foreground.shape", self.artificial_foreground.shape))
                                self.__fit_borders(w, h)
                                foreground = cv2.bitwise_and(self.artificial_foreground, self.all_masks[ind])
                                self.all_foregrounds[ind] = foreground
                                # todo: create artificial mask when transforming the artificial image
                                a_height, a_width = self.artificial_foreground.shape[:2]  # params['image'].shape[:2]
                                artificial_mask = None

                        # else:
                        #     self.coords.append(coord)


def artificate(image, cnts, artifical_background=None, foregrounds=None, masks=None):
    # GUI_interaction.imgshow(artfical)
    # GUI_interaction.imgshow(image)
    real_image = image.copy()
    artifical_image = image.copy()
    if foregrounds and masks:
        for ind, contour in cnts.items():
            x, y, w, h = contour
            mask = masks[ind]
            logging.debug(("mask", mask))
            artifical = real_image[y:y + h, x:x + w]
            # GUI_interaction.imgshow(artifical)
            logging.debug(("artifical", artifical))
            mask_p = cv2.bitwise_not(mask)
            logging.debug(("mask_p", mask_p))
            roi = cv2.bitwise_and(artifical, mask_p)
            logging.debug(("roi.shape", roi.shape))
            if not isinstance(foregrounds[ind], type(None)):
                logging.debug(("foregrounds[ind]", foregrounds[ind]))
                logging.debug(("foregrounds[ind].shape", foregrounds[ind].shape))
                artifical_image[y:y + h, x:x + w] = cv2.bitwise_or(roi, foregrounds[ind])
                # real_image = image
            # print(x,y,w,h)
        if not isinstance(artifical_background, type(None)):
            for ind, contour in cnts.items():
                x, y, w, h = contour
                artifical = artifical_image[y:y + h, x:x + w]
                # foreground = cv2.bitwise_and(artifical, mask)
                artifical_background[y:y + h, x:x + w] = artifical  # cv2.bitwise_or(roi, foreground)
            artifical_image = artifical_background
            # real_image = artifical_background

    else:
        for contour in cnts.values():
            x, y, w, h = contour
            if not isinstance(artifical_background, type(None)):
                artifical_background[y:y + h, x:x + w] = real_image[y:y + h, x:x + w]
                real_image = artifical_background
        # GUI_interaction.imgshow(artfical)
    return artifical_image


def artificate_foreground(image, artfical, cnts):
    # GUI_interaction.imgshow(artfical)
    # GUI_interaction.imgshow(image)
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        artfical[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        # GUI_interaction.imgshow(artfical)

    return artfical


def process_with_check(path_to_store, path_to_save, foreground_path=None, background_path=None,
                       mask_ext=None, name_val_match=None):
    """

    :param path_to_store: where the initial images stored
    :param path_to_save: where to save augmented images
    :param foreground_path: where foregrounds stored
    :param background_path: where backgrounds stored
    :param mask_ext: mask extentions
    :param name_val_match: foreground name - mask value match
    :return:
    """
    mask_path = os.path.join(path_to_store, "mask")
    image_path = os.path.join(path_to_store, "image")
    filelist_box = FolderIterator.get_file_list(image_path, full_path=False)
    wrong = []
    len(filelist_box)

    for path in tqdm(filelist_box):
        # print(mask_path, path)
        mask_name = PathSupport.takename(path)[0]
        if mask_ext:
            mask_name = mask_name + mask_ext
        else:
            mask_name = path
        image = IO.read(os.path.join(image_path, path))
        mask = IO.read(os.path.join(mask_path, mask_name), 0)
        h, w = image.shape[:2]
        # if image.shape[:2] != mask.shape[:2]:
        #     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #     h, w = image.shape[:2]

        if image.shape[:2] != mask.shape[:2]:
            wrong.append(path)

        else:
            if w > h:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            num = 0
            ia = ImageArtification_self(foreground_folder_path=foreground_path, background_folder_path=background_path,
                                        name_val_match=name_val_match)
            ia.get_params_dependent_on_targets(image, mask)
            artfical = artificate(image=ia.image, foregrounds=ia.all_foregrounds,
                                  artifical_background=ia.artificial_background,
                                  cnts=ia.coord_d, masks=ia.all_masks)
            #

            aug = AlbumentationOptions.box_segmentation_aug()
            augment = aug(image=artfical, mask=mask)
            artfical = augment["image"]
            mask = augment["mask"]


            name = os.path.split(path)[-1]
            new_name = PathSupport.takename(name)[0]
            ext = PathSupport.takeext(name)[0]
            mask_ext = PathSupport.takeext(mask_name)[0]
            name_fin = new_name + "-" + str(num)
            print(os.path.join(path_to_save, "image", name))
            assert IO.write(artfical, os.path.join(path_to_save, "image", name_fin + ext)), \
                "save path is wrong " + os.path.join(path_to_save, "image", name_fin + ext)
            assert IO.write(mask, os.path.join(path_to_save, "mask", name_fin + mask_ext)), \
                "save path is wrong " + os.path.join(path_to_save, "mask", name_fin + mask_ext)
            # print(path)
            shutil.move(os.path.join(image_path, path), os.path.join("D:/Local drive/Segmentation task/All_data/done", name))
            num += 1

    if len(wrong) > 0:
        print("These images where ignored", wrong)


if __name__ == "__main__":
    Debugger("aug_test")
    path_for_segm_labels = "D:/OneDrive/Skoltech/Projects/Pythons_project/Database/Processed_data/Segmentation/Box_segm.json"
    segclass = Segclass_json_transform(
        path_for_segm_labels)  # ignore=["limestone", "static", "unlabeled", "core_column"] ,"static","cracks" ignore=["limestone", "static", "unlabeled"]
    segclass.load_label_ids()
    print(segclass.__info__())
    process_with_check(path_to_store="D:/Local drive/Segmentation task/All_data/train/train",#initial",#
                       path_to_save="D:/Local drive/Segmentation task/aug_All_data",#",#
                       foreground_path="D:/Local drive/Segmentation task/foreground",
                       background_path="D:/Local drive/Segmentation task/background",#"D:/OneDrive/Skoltech/Projects/Pythons_project/Database/Processed_data/augs/background",
                        name_val_match=segclass.name_id)#mask_ext=".png",
