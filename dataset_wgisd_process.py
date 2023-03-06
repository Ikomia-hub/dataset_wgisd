import os
import copy
import numpy as np
from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dataset
import cv2


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class DatasetWgisdParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.data_folder_path = ""
        self.class_file_path = ""
        self.seg_mask_mode = "None"

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.data_folder_path = params["data_folder_path"]
        self.class_file_path = params["class_file_path"]
        self.seg_mask_mode = params["seg_mask_mode"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "data_folder_path": self.data_folder_path,
            "class_file_path": self.class_file_path,
            "seg_mask_mode": self.seg_mask_mode,
        }
        return params


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class DatasetWgisd(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the process here
        self.add_output(datasetio.IkDatasetIO("yolo"))
        self.add_output(dataprocess.CNumericIO())

        # Create parameters class
        if param is None:
            self.set_param_object(DatasetWgisdParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def load_masks(self, data):
        print("Generating masks for semantic segmentation. This may take a while, please wait...")
        images_with_mask = []
        param = self.get_param_object()

        for image in data["images"]:
            filename, extension = os.path.splitext(image["filename"])
            mask_file = filename + ".npz"

            if os.path.isfile(mask_file):
                if param.seg_mask_mode == "Instance":
                    image["instance_seg_masks_file"] = mask_file
                else:
                    mask_file_png = filename + "mask" + ".png"
                    if not os.path.isfile(mask_file_png):
                        d = np.load(mask_file)
                        mask = d["arr_0"]
                        mask = np.max(mask, axis=-1)
                        cv2.imwrite(mask_file_png, mask)

                    image["semantic_seg_masks_file"] = mask_file_png

                images_with_mask.append(image)

        data["images"] = images_with_mask

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get dataset output :
        output = self.get_output(0)
        output.data = dataset.load_yolo_dataset(param.data_folder_path, param.class_file_path)
        output.has_bckgnd_class = False
        # Step progress bar:
        self.emit_step_progress()

        if param.seg_mask_mode != "None":
            self.load_masks(output.data)

        # Step progress bar:
        self.emit_step_progress()

        # Class labels output
        numeric_out = self.get_output(1)
        numeric_out.clear_data()
        numeric_out.set_output_type(dataprocess.NumericOutputType.TABLE)

        class_ids = []
        for i in range(len(output.data["metadata"]["category_names"])):
            class_ids.append(i)

        numeric_out.add_value_list(class_ids, "Id", list(output.data["metadata"]["category_names"].values()))

        # Step progress bar:
        self.emit_step_progress()

        # Call endTaskRun to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class DatasetWgisdFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "dataset_wgisd"
        self.info.short_description = "Load Wine Grape Instance Segmentation Dataset (WGISD)"
        self.info.description = "Load Wine Grape Instance Segmentation Dataset (WGISD). " \
                                "This dataset was created to provide images and annotations " \
                                "to study object detection, instance or semantic segmentation " \
                                "for image-based monitoring and field robotics in viticulture. " \
                                "It provides instances from five different grape varieties taken on field. " \
                                "These instances shows variance in grape pose, illumination and focus, " \
                                "including genetic and phenological variations such as shape, color and compactness."
        self.info.authors = "Thiago T. Santos, Leonardo L. de Souza, Andreza A. dos Santos, Sandra Avila"
        self.info.article = "Grape detection, segmentation, and tracking using deep neural networks " \
                            "and three-dimensional association"
        self.info.year = 2020
        self.info.license = "CC BY-NC 4.0"
        self.info.documentation_link = "https://github.com/thsant/wgisd"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-dev/WGISD_Dataset"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Dataset"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/wgisd.png"
        self.info.keywords = "dataset,annotation,train,dnn"

    def create(self, param=None):
        # Create process object
        return DatasetWgisd(self.info.name, param)
