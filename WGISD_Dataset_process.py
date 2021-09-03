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
class WGISD_DatasetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.data_folder_path = ""
        self.class_file_path = ""
        self.seg_mask_mode = "None"

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.data_folder_path = param_map["data_folder_path"]
        self.class_file_path = param_map["class_file_path"]
        self.seg_mask_mode = param_map["seg_mask_mode"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["data_folder_path"] = self.data_folder_path
        param_map["class_file_path"] = self.class_file_path
        param_map["seg_mask_mode"] = self.seg_mask_mode
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class WGISD_DatasetProcess(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(datasetio.IkDatasetIO("yolo"))
        self.addOutput(dataprocess.CNumericIO())

        # Create parameters class
        if param is None:
            self.setParam(WGISD_DatasetParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def load_masks(self, data):
        print("Generating masks for semantic segmentation. This may take a while, please wait...")
        images_with_mask = []
        param = self.getParam()

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
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get dataset output :
        output = self.getOutput(0)
        output.data = dataset.load_yolo_dataset(param.data_folder_path, param.class_file_path)
        output.has_bckgnd_class = False
        # Step progress bar:
        self.emitStepProgress()

        if param.seg_mask_mode != "None":
            self.load_masks(output.data)

        # Step progress bar:
        self.emitStepProgress()

        # Class labels output
        numeric_out = self.getOutput(1)
        numeric_out.clearData()
        numeric_out.setOutputType(dataprocess.NumericOutputType.TABLE)

        class_ids = []
        for i in range(len(output.data["metadata"]["category_names"])):
            class_ids.append(i)

        numeric_out.addValueList(class_ids, "Id", list(output.data["metadata"]["category_names"].values()))

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class WGISD_DatasetProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "WGISD_Dataset"
        self.info.shortDescription = "Load Wine Grape Instance Segmentation Dataset (WGISD)"
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
        self.info.documentationLink = "https://github.com/thsant/wgisd"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-dev/WGISD_Dataset"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Dataset"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/wgisd.png"
        self.info.keywords = "dataset,annotation,train,dnn"

    def create(self, param=None):
        # Create process object
        return WGISD_DatasetProcess(self.info.name, param)
