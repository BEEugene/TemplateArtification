import logging
import numpy as np
import cv2

from utilities.filesystem import PathSupport


class IO:

    def __init__(self):
        self.logger = logging.getLogger("IO")

    @classmethod
    def read(cls, impath, flag=cv2.IMREAD_COLOR, resize=None):
        """
        Read an image with full path
        :param impath: full path to the image
        :param flag:
        :param resize:
        :type : int
        :return:
        """
        logger = logging.getLogger("IO.read")
        logger.debug(("impath", impath))
        logger.debug(("flag", flag))
        try:
            image = cv2.imread(impath, flag)
            assert image is not None, ("Image is {im} check the path {path}".format (im=image, path=impath))
        except Exception as e:
            print("Some problems with path, this error happen: ", e)
            image = cv2.imdecode(np.fromfile(impath, dtype=np.uint8), flag)
            assert image is not None, ("Image is {im} check the path {path}".format(im=image, path=impath))
        if resize:
            image = cv2.resize(image, tuple(int(i/resize) for i in reversed(image.shape[:2])))
        logger.debug(("image.shape", image.shape))
        return image

    @classmethod
    def write(cls, image, impath, jpg_quality=None, flag=-1):
        # print("impath.encode(\"utf-8\")", impath.encode("utf-8"))
        ext = PathSupport.takeext(impath)[0]
        try:

            impath.encode("latin1") # test if there unicde in the path
            if jpg_quality and ext.lower() == ".jpg":
                result = cv2.imwrite(impath, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                result = cv2.imwrite(impath, image)
        except Exception as e:
            print("Some problems with path, this error happen: ", e)

            assert ext.lower() == ".jpg" or ext.lower() ==".tiff" or ext.lower() == ".png", ("The extention is wrong: ", ext)
            if jpg_quality and ext.lower() == ".jpg":
                result, im_buf_arr = cv2.imencode(ext, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                result, im_buf_arr = cv2.imencode(ext, image)#"."+PathSupport.takeext(impath)[0]
            # print("impath",impath)
            im_buf_arr.tofile(impath)
        print("saved to", impath)
        if not cv2.os.path.isfile(impath):
            # cv2.imshow('res', image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            result = None

        assert not isinstance(result, type(None)), ("Image is {im} check the path {path}".format(im=image, path=impath))
        return result