import logging
import os
import re

from logger.logparams import Debug_param


class PathSupport:

    @staticmethod
    def takenameext(aby):
        if isinstance(aby,str):
            aby=[aby]
        return [os.path.split(each)[-1] for each in aby]

    @staticmethod
    def takename(any):
        if isinstance(any,str):
            any=[any]
        return [os.path.splitext(each_of)[0] for each_of in PathSupport.takenameext(any)]

    @staticmethod
    def takeext(any):
        if isinstance(any,str):
            any=[any]
        return [os.path.splitext(each_of)[1] for each_of in PathSupport.takenameext(any)]

class FolderIterator:

    def __init__(self):
        pass

    @staticmethod
    def get_file_list(folder, extension=["any"], full_path=False):
        """
        get the files without any folders
        :param folder: which folder to list
        :type: str or list
        :param extension: which extenstion
        :type: list
        :param full_path:
        :return:
        """
        logger = logging.getLogger("get_file_list")
        logger.setLevel(Debug_param.debug_scope())

        if extension == ["any"]:
            extension=["."]
        file_list = []
        for file in os.listdir(folder):
            logger.info(("file is ", file))
            logger.info(("folder is ", folder))
            if os.path.isfile(os.path.join(folder, file)):
                assert isinstance(extension, list), "Provide a list into extension!"
                for each in extension:
                    logger.info(("extension is ", each))
                    logger.info(("check is", each in file))
                    if re.search(each, file, re.IGNORECASE):
                        if full_path:
                            file_list.append(os.path.join(folder,file))
                        else:
                            file_list.append(file)
        logger.debug(("file_list is ", file_list))
        if file_list == []:
            print("Warning! NO files in the list, check the extension ({})  or path ({})".format(extension, folder))
        return file_list

    @staticmethod
    def get_folder_list(paths, full_path=False):
        logger = logging.getLogger("get_folder_list")
        logger.setLevel(logging.INFO)
        folder_list = []
        logger.debug(("paths are ", paths))
        assert isinstance(paths, list), "Provide a list into paths!"
        for path in paths:
            logger.debug(("path is ", path))
            for folder in os.listdir(path):
                logger.debug(("folder is ", folder))
                if os.path.isdir(os.path.join(path, folder)):
                    if full_path:
                        folder_list.append(os.path.join(path, folder))
                    else:
                        folder_list.append(folder)
        logger.debug(("folder_list is ", folder_list))
        if folder_list == []:
            print("Warning! NO folders in the list, check the path ({})".format(paths))
        return folder_list

    @staticmethod
    def sort_path_list(pathlist):
        key = PathSupport.takename
        return sorted(list(pathlist), key=key)
