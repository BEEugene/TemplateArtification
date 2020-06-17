import json
import logging

from logger.logparams import Debug_param


class Segclass_json_transform:
    # logger = logging.getLogger("Segclass_json_transform")
    # logger.setLevel(logging.DEBUG)
    """"""

    def __init__(self, path_to_json, ignore=[], not_to_ignore=[]):
        """

        :param path_to_json: name_val_match path
        :param ignore: the list of labels which should not be loaded
        :param not_to_ignore: the list of labels which should be loaded
        :store
        """
        with open(path_to_json) as f:
            self.json = json.load(f)
            self.ignore = ignore
            self.not_ignore = not_to_ignore
            self.work_with = self.json["labels"]
            self.labels = self.work_with.keys()
            self.id_labels = {}
            self.color_labels = {}
            self.name_id = {}
            self.numbered_labels = {}
            self.num_id_map = {}
        self.logger = logging.getLogger("Segclass_json_transform")
        self.logger.setLevel(Debug_param.debug_scope())
        # self.logger.info(("\n"+"{}\nScript is runing\n{}".format("=" * 50, "=" * 50)))

    def __info__(self):
        return {
            "values of seg_class":
                {
                    "self.ignore:": self.ignore,
                    "self.labels:": self.labels,
                    "self.id_labels:": self.id_labels,
                    "self.color_labels:": self.color_labels,
                    "self.name_id:": self.name_id,
                    "self.numbered_labels,:": self.numbered_labels,
                    "self.num_id_map:": self.num_id_map}
            # "\n", "self.name_val_match:", self.name_val_match,
            # "\n", "self.work_with,:": self.work_with,
        }

    def load_label_ids(self):

        number = 1

        if self.not_ignore != []:
            for item in self.work_with:
                if item not in self.not_ignore:
                    self.ignore.append(item)

        for label in self.labels:
            if label in self.ignore:
                print("Label", label, "is ignored.")
            else:
                id = self.work_with[label]["id"]
                self.id_labels[id] = label
                self.name_id[label] = id
                if id is 0:
                    number_cache = number  # save the number to cache
                    number = 0
                self.numbered_labels[number] = label
                self.num_id_map[id] = number
                if id is 0:
                    number = number_cache  # restore the cached number
                number += 1

        self.logger.debug((self.__info__()))

        return self.id_labels, self.name_id  # , self.numbered_labels, self.num_id_map

    def load_label_colors(self):
        """creates mapping to color labels from name_val_match data
        like {33: [213, 62, 7], 15: [255, 255, 0], 18: [160, 110, 148]}"""
        # self.logger = logging.getLogger("Segclass_json_transform_load_label_colors")
        if self.id_labels == {}:
            self.load_label_ids()

        for id, label in self.id_labels.items():
            if label not in self.ignore:
                color = self.work_with[label]["color"]
                self.color_labels[id] = color

        self.logger.debug((self.__info__()))

        return self.color_labels