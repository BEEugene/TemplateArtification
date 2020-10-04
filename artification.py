import logging
import os
import random
import shutil

import imutils
import numpy as np
import cv2
from tqdm import tqdm

from augmentationoptions.albumentationsoptions import AlbumentationOptions
from logger.logparams import Debugger
from utilities.filesystem import FolderIterator, PathSupport
from utilities.inputoutput import IO
from utilities.jsonprocessor import JsonParser


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

    def __init__(self, background_folder_path:str=None, foreground_folder_path:str=None,foreground_mask_path:str=None,
                 flag=cv2.MIXED_CLONE, bg_param=None, fg_param=None, name_val_match=None,
                 always_apply=True, p_b=0, p_f=1, p_f_freq=.7):
        """

        :param background_folder_path:
        :param foreground_folder_path:
        :param flag:
        :param bg_param:
        :param fg_param:
        :param name_val_match:
        :param always_apply:
        :param p_b: background change probability
        :param p_f: foreground change probability
        :param p_f_freq: foreground change probability
        """
        self.logger = logging.getLogger("ImageArtification_self")
        self.background_folder = background_folder_path
        self.foreground_folder = foreground_folder_path
        self.foreground_mask_path = foreground_mask_path
        self.proc_params(bg_param, fg_param)
        self.name_val_match = name_val_match
        self.masks = []
        self.all_masks = {}
        self.all_foregrounds = {}
        self.all_foreground_masks = {}
        self.proc_folders()
        self.cnts = None
        self.coords = []
        self.artificial_background = None
        self.artificial_foreground = None
        self.foreground_mask = None
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
        self.chosen_fg = None
        self.chosen_bg = None

    def proc_params(self, bg_param, fg_param):
        if not bg_param:
            self.bg_param = {}
        else:
            self.bg_param = bg_param
        if not fg_param:
            self.fg_param = {}
        else:
            self.fg_param = fg_param

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

    def __aug_bg_back(self):
        if self.arti_height < self.im_height or self.arti_width < self.im_width:
            clac_dif_h = self.im_height - self.arti_height
            clac_dif_w = self.im_width - self.arti_width
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
        return self.artificial_background


    def __aug_bg_cov(self):

        ratio = self.bg_param["ratio"]

        self.artificial_background = imutils.resize(self.artificial_background,
                                                 self.im_height//ratio)

        return self.artificial_background

    def __aug_fg_back(self, height, width):
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
            if not isinstance(self.foreground_mask, type(None)):
                self.foreground_mask = cv2.resize(self.foreground_mask, (width, height))
        if self.arti_height > height or self.arti_width > width:
            self.artificial_foreground = self.artificial_foreground[:height, :width, :]
            if not isinstance(self.foreground_mask, type(None)):
                self.foreground_mask = self.foreground_mask[:height, :width]

    def __fit_borders(self, width=None, height=None):
        # transform = albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True) # didn't work
        # new=transform.apply(self.artificial_image)
        if not isinstance(self.artificial_background, type(None)):

            if self.bg_param["bottom"]:
                self.artificial_background = self.__aug_bg_back()
            else:
                self.artificial_background = self.__aug_bg_cov()

        if not isinstance(self.artificial_foreground, type(None)) and height and width:
            self.__aug_fg_back(width=width, height=height)

    def __choose_object(self, chosen):
        logger = self.logger.getChild("__chosen_obgect")
        chosen_path = os.path.join(self.background_folder, chosen)
        chosen_object = random.choice(os.listdir(chosen_path))
        artificial_image_path = os.path.join(chosen_path, chosen_object)
        logger.debug(("artificial_image_path", artificial_image_path))
        logger.debug(("chosen_path", chosen_path))
        return artificial_image_path
    # def __process_background_params(self, background):

    def get_params_dependent_on_targets(self, image, mask):

        logger = self.logger.getChild("get_params_dependent_on_targets")
        self.image = image
        logger.debug(("self.image.shape", self.image.shape))
        self.im_height, self.im_width = self.image.shape[:2]  # params['image'].shape[:2]
        self.cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(self.cnts))
        coords = []

        if self.foreground_folder:
            self.chosen_fg = random.choice(self.fg_folds)
        if self.background_folder:
            self.chosen_bg = random.choice(self.bg_folds)
            self.bg_param = self.bg_param[self.chosen_bg]
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
            self.all_foreground_masks[ind] = None
            logger.debug(("mask_final.shape", mask_final.shape))
            logger.debug(("median", median))
            logger.debug(("median", median))
            logger.debug(("self.name_val_match[chosen_fg] ", self.name_val_match[self.chosen_fg]))

        if self.background_folder and random.random() < self.p_b:
            chosen_path = os.path.join(self.background_folder, self.chosen_bg)
            chosen_object = random.choice(os.listdir(chosen_path))
            artificial_image_path = os.path.join(chosen_path, chosen_object)
            logger.debug(("artificial_image_path", artificial_image_path))
            logger.debug(("chosen_path", chosen_path))
            logger.debug(("chosen_bg", self.chosen_bg))
            bg_check = not self.bg_param
            if bg_check:
                self.artificial_background = IO.read(artificial_image_path)
            elif self.bg_param["search_mask"]:
                self.artificial_background = IO.read(artificial_image_path, flag=-1)
                if self.artificial_background.shape[-1]>3:
                    bg_im_mask = self.artificial_background[:,:,3]
                    self.artificial_background = cv2.bitwise_and(self.artificial_background[:,:,:3],
                                                                 cv2.merge((bg_im_mask,bg_im_mask,bg_im_mask)))
            else:
                logger.debug(("the mask wil not be searched, this is a background image"))
                self.artificial_background = IO.read(artificial_image_path)
            try:
                # self.artificial_background = IO.read(artificial_image_path)
                self.arti_height, self.arti_width = self.artificial_background.shape[:2]
            except Exception as e:
                print("e", e)
                print(artificial_image_path)
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
                if self.chosen_fg in names:
                    for ind, coord in self.coord_d.items():
                        if medians[ind] == self.name_val_match[self.chosen_fg]:
                            x, y, w, h = coord
                            if random.random() < self.p_f_freq:
                                chosen_path = os.path.join(self.foreground_folder, self.chosen_fg)
                                chosen_object = random.choice(os.listdir(chosen_path))
                                artificial_image_path = os.path.join(chosen_path, chosen_object)
                                logger.debug(("artificial_image_path", artificial_image_path))
                                logger.debug(("chosen_path", chosen_path))
                                self.artificial_foreground = IO.read(artificial_image_path)
                                self.arti_height, self.arti_width = self.artificial_foreground.shape[:2]
                                logger.debug(("self.arti_height < self.im_height or self.arti_width < self.im_width",
                                              self.arti_height < self.im_height or self.arti_width < self.im_width))
                                logger.debug(("self.artificial_foreground.shape", self.artificial_foreground.shape))

                                if self.arti_height*2 < h: #if the artificial height is larger then original height
                                    foreground = None
                                else:
                                    if self.foreground_mask_path:
                                        chosen_mask_path = os.path.join(self.foreground_mask_path, self.chosen_fg)
                                        if os.path.exists(chosen_mask_path):
                                            mask_list = os.listdir(chosen_mask_path)
                                            cur_obj_name = PathSupport.takename(chosen_object)[0]
                                            find_mask = [each for each in filter(lambda x: x!=None,
                                                                                 [each if cur_obj_name in each else None
                                                                                  for each in mask_list])]
                                            if len(find_mask) == 1:
                                                self.foreground_mask = IO.read(os.path.join(chosen_mask_path,
                                                                                                  find_mask[0]),
                                                                               flag=cv2.IMREAD_GRAYSCALE)
                                            else:
                                                self.foreground_mask = None

                                    self.__fit_borders(w, h)
                                    foreground = cv2.bitwise_and(self.artificial_foreground, self.all_masks[ind])
                                    self.all_foreground_masks[ind] = self.foreground_mask
                                self.all_foregrounds[ind] = foreground

                                # todo: create artificial mask when transforming the artificial image
                                a_height, a_width = self.artificial_foreground.shape[:2]  # params['image'].shape[:2]
                                artificial_mask = None

                        # else:
                        #     self.coords.append(coord)


def artificate(image, cnts, artifical_background=None, foregrounds=None,foreground_masks=None,
               masks=None, background_type=None, configs=None, ini_mask=None):
    # GUI_interaction.imgshow(artfical)
    # GUI_interaction.imgshow(image)
    logger = logging.getLogger("artificate")
    real_image = image.copy()
    artifical_image = image.copy()
    if foregrounds and masks:
        for ind, contour in cnts.items():
            # replace real image part with a artificial one
            # e.g. take roi with core on the real image and change it to other core
            x, y, w, h = contour
            mask = masks[ind]
            logger.debug(("mask", mask))
            artifical = real_image[y:y + h, x:x + w]
            # GUI_interaction.imgshow(artifical)
            logger.debug(("artifical", artifical))
            mask_p = cv2.bitwise_not(mask)
            logger.debug(("mask_p", mask_p))
            roi = cv2.bitwise_and(artifical, mask_p)
            logger.debug(("roi.shape", roi.shape))
            if not isinstance(foregrounds[ind], type(None)):
                logger.debug(("foregrounds[ind]", foregrounds[ind]))
                logger.debug(("foregrounds[ind].shape", foregrounds[ind].shape))
                artifical_image[y:y + h, x:x + w] = cv2.bitwise_or(roi, foregrounds[ind])
                if not isinstance(foreground_masks[ind], type(None)):
                    ini_mask[y:y + h, x:x + w] = cv2.bitwise_and(foreground_masks[ind], mask[:, :, 0])
                # real_image = image
            # print(x,y,w,h)
        if not isinstance(artifical_background, type(None)):
            bottom_check = configs[background_type]["bottom"]
            logger.debug(("bottom_check", background_type, configs[background_type]["bottom"]))
            temp_image = artifical_image.copy()
            for ind, contour in cnts.items():
                x, y, w, h = contour
                if bottom_check: # should it be on the background or is it a foreground object
                    artifical = artifical_image[y:y + h, x:x + w]
                    # foreground = cv2.bitwise_and(artifical, mask)
                    artifical_background[y:y + h, x:x + w] = artifical  # cv2.bitwise_or(roi, foreground)
                    temp_image = artifical_background

            artifical_image = temp_image
                # else: # if it is a foreground object
            if not bottom_check:
                b_h, b_w = artifical_background.shape[:2]
                a_h, a_w = artifical_image.shape[:2]
                x = random.randrange(0, a_w - (b_w//2))
                y = random.randrange(0, a_h - (b_h//2))
                # if roi.shape[:2] >
                tot_height = y + b_h
                tot_width = x + b_w
                if tot_height > a_h:
                    fin_h = a_h - tot_height
                    fin_h = tot_height + fin_h
                else:
                    fin_h = tot_height
                if tot_width > a_w:
                    fin_w = a_w - tot_width
                    fin_w = tot_width + fin_w
                else:
                    fin_w = tot_width
                roi = artifical_image[y:fin_h, x:fin_w].copy()
                roi_h, roi_w = roi.shape[:2]
                artifical_background = artifical_background[:roi_h, :roi_w, :]
                mask = artifical_background.copy()
                mask[mask > 0] = 255
                mask = cv2.bitwise_not(mask)
                roi = cv2.bitwise_and(roi, mask)

                new_im = cv2.bitwise_or(roi, artifical_background)
                artifical_image[y:fin_h, x:fin_w] = new_im
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                roi = ini_mask[y:fin_h, x:fin_w].copy()
                new_im = cv2.bitwise_and(roi, mask)
                ini_mask[y:fin_h, x:fin_w] = new_im
                # ini_mask = cv2.bitwise_not(ini_mask)


            artifical_image = artifical_image
            # real_image = artifical_background
            return artifical_image, ini_mask

    else:
        if not isinstance(artifical_background, type(None)):
            for contour in cnts.values():
                x, y, w, h = contour
                artifical_background[y:y + h, x:x + w] = real_image[y:y + h, x:x + w]
                artifical_image = artifical_background
        # GUI_interaction.imgshow(artfical)
    return artifical_image, None


def artificate_foreground(image, artfical, cnts):
    # GUI_interaction.imgshow(artfical)
    # GUI_interaction.imgshow(image)
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        artfical[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        # GUI_interaction.imgshow(artfical)

    return artfical


def process_with_check(path_to_store: str, path_to_save: str, foreground_path: str=None, background_path: str = None,
                       foreground_mask_path:str = None,
                       mask_ext: str = None, image_ext: str = None, name_val_match: dict = None, background_params: dict = None,
                       mask_path: str = "mask", image_path: str = "image"):
    """

    :param path_to_store: where the initial images stored
    :param path_to_save: where to save augmented images
    :param foreground_path: where foregrounds stored
    :param background_path: where backgrounds stored
    :param mask_ext: mask extentions
    :param name_val_match: foreground name - mask value match
    :return:
    """
    logger = logging.getLogger("process_with_check")
    mask_path = os.path.join(path_to_store, mask_path)
    image_path = os.path.join(path_to_store, image_path)
    filelist_box = FolderIterator.get_file_list(mask_path, full_path=False)
    wrong = []
    # len(filelist_box)

    logger.debug(("mask_path", mask_path))
    logger.debug(("image_path", image_path))
    logger.debug(("filelist_box", len(filelist_box), filelist_box))
    for path in tqdm(filelist_box):
        # print(mask_path, path)
        name = PathSupport.takename(path)[0]
        if mask_ext:
            mask_name = name + mask_ext
        else:
            mask_name = path
        if image_ext:
            image_name = name + image_ext
        else:
            image_name = path
        if os.path.exists(os.path.join(image_path, image_name)):
            image = IO.read(os.path.join(image_path, image_name))
            mask = IO.read(os.path.join(mask_path, mask_name), 0)
            h, w = image.shape[:2]

            if image.shape[:2] != mask.shape[:2]:
                logger.debug(("image_name", image_name))
                logger.debug(("Shapes are not equal!"))
                wrong.append(path)

            # if image.shape[:2] != mask.shape[:2]:
            #     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            #     h, w = image.shape[:2]



            else:
                if w > h:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                num = 0
                ia = ImageArtification_self(foreground_folder_path=foreground_path, background_folder_path=background_path,
                                            foreground_mask_path = foreground_mask_path,
                                            name_val_match=name_val_match,bg_param=background_params)
                ia.get_params_dependent_on_targets(image, mask)
                artfical, _ = artificate(image=ia.image, foregrounds=ia.all_foregrounds,
                                         foreground_masks=ia.all_foreground_masks,
                                      artifical_background=ia.artificial_background,
                                      cnts=ia.coord_d, masks=ia.all_masks, ini_mask=mask,
                                         background_type=ia.chosen_bg,
                                         configs=background_params)
                #
                if not isinstance(_, type(None)):
                    mask = _
                # aug = AlbumentationOptions.box_segmentation_aug()
                # augment = aug(image=artfical, mask=mask)
                # artfical = augment["image"]
                # mask = augment["mask"]


                name = os.path.split(path)[-1]
                new_name = PathSupport.takename(name)[0]
                # ext = PathSupport.takeext(name)[0]
                mask_ext = PathSupport.takeext(name)[0]
                name_fin = new_name + "-" + str(num)
                # print(os.path.join(path_to_save, "image", name))
                mask_save_path = os.path.join(path_to_save, "mask")
                image_save_path = os.path.join(path_to_save, "image")
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                if not os.path.exists(image_save_path):
                    os.makedirs(image_save_path)
                assert IO.write(artfical, os.path.join(image_save_path, name_fin + image_ext)), \
                    "save path is wrong " + os.path.join(image_save_path, name_fin + image_ext)
                assert IO.write(mask, os.path.join(mask_save_path, name_fin + mask_ext)), \
                    "save path is wrong " + os.path.join(mask_save_path, name_fin + mask_ext)
                # print(path)
                # shutil.move(os.path.join(image_path, path), os.path.join("D:/Local drive/Segmentation task/All_data/done", name))
                num += 1
        else:
            logger.debug(("path doesn't exists", os.path.join(image_path, image_name)))
            wrong.append(path)

    if len(wrong) > 0:
        print("These images were ignored", wrong)


if __name__ == "__main__":
    Debugger("aug_test")
    path_for_segm_labels = "./examples/Box_segmentation.json"
    segclass = JsonParser(
        path_for_segm_labels)
    segclass.load_label_ids()
    print(segclass.__info__())
    background_parameters = {"ground":{"bottom": True, "search_mask": False,
                                       "cover_foreground": False},
                             "hammer":{"bottom": False, "cover_foreground": True,
                                       "search_mask": True, "ratio":5},
                             "pen": {"bottom": False, "cover_foreground": True,
                                        "search_mask": True, "ratio": 22},
                             "hat":{"bottom": False, "cover_foreground": True,
                                       "search_mask": True, "ratio":5},
                             "hand": {"bottom": False, "cover_foreground": True,
                                        "search_mask": True, "ratio": 9},
                             "ruler":{"bottom": False, "cover_foreground": True,
                                       "search_mask": True, "ratio":11},

                             }

    main_path = "D:/Local drive/Segmentation task/Что сделано"

    subpaths = [
                "./01032020",
                "./05032020",
                "./15042020",
                "./What I've done/1",
                "./What I've done/2",
                "./What I've done/5",
                "./What I've done/6",
                "./What I've done/7",
                "./What I've done/8",
                "./What I've done/9",
                "./What I've done/11",
                "./What I've done/12",
                "./29022020 что сделано/Большая площадь, скв. 297, Ящ. 1-12",
                "./29022020 что сделано/Галяновская площадь, скв. 2631, Ящ. 1-13",
                "./29022020 что сделано/Загадочные 86 коробок",
                "./29022020 что сделано/Ольховское, скв. 301, Ящ. 1-9",
                "./29022020 что сделано/Средне-Назымское, скв. 311, Ящ. 1-18",
                "./Скв. 888-06, ящ. 1-40/"]
    image_path = "image"#"image/DL"
    # mask_path = "mask"
    # for path in tqdm(subpaths):
    storage = "D:/OneDrive/Skoltech/Projects/Pythons_project/Database/Processed_data/augs/core_background/empty_create"  #os.path.join(main_path, path)# where images stored
    assert os.path.exists(storage), ("Path doesn't exist:", storage)
    # print(storage, os.path.exists(storage))
    save_store = os.path.join("D:/Local drive/Segmentation task/augmented_new", "test")
    if not os.path.exists(save_store):
        os.makedirs(save_store)
    process_with_check(path_to_store=storage, #"./examples/data_sample",#initial",#
                       path_to_save=save_store,#"D:/Local drive/Pycharm/TemplateArtification/examples/aug_sample",#",#
                       foreground_path= "C:/Users/ebara/Documents/extracts/image/",#"D:/Local drive/Segmentation task/foreground",
                       foreground_mask_path= "C:/Users/ebara/Documents/extracts/mask/",#"D:/Local drive/Segmentation task/foreground_mask",
                       background_path="D:/OneDrive/Skoltech/Projects/Pythons_project/Database/Processed_data/augs/background",#"./examples/backgrounds",#"D:/OneDrive/Skoltech/Projects/Pythons_project/Database/Processed_data/augs/background",
                        name_val_match=segclass.name_id, background_params=background_parameters,
                       image_ext=".jpg", image_path=image_path)#mask_ext=".png",
