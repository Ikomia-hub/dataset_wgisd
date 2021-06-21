from ikomia import core, dataprocess
from ikomia.dnn import datasetio, dataset
import copy
# Your imports below
import os
import numpy as np
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
        self.load_mask = True

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.data_folder_path = paramMap["data_folder_path"]
        self.class_file_path = paramMap["class_file_path"]
        self.load_mask = bool(paramMap["load_mask"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = core.ParamMap()
        paramMap["data_folder_path"] = self.data_folder_path
        paramMap["class_file_path"] = self.class_file_path
        paramMap["load_mask"] = str(self.load_mask)
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class WGISD_DatasetProcess(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(datasetio.IkDatasetIO("yolo"))
        self.addOutput(dataprocess.CDblFeatureIO())

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
        images_with_mask = []

        for image in data["images"]:
            filename, extension = os.path.splitext(image["filename"])
            mask_file = filename + ".npz"

            if os.path.isfile(mask_file):
                mask_file_png = filename + "mask" + ".png"
                if not os.path.isfile(mask_file_png):
                    d = np.load(mask_file)
                    mask = d["arr_0"]
                    mask = np.max(mask, axis=-1)
                    cv2.imwrite(mask_file_png, mask)
                image["instance_seg_masks_file"] = mask_file_png
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

        if param.load_mask:
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
        self.info.shortDescription = "Load WGSID dataset"
        self.info.description = "Load YOLO dataset from a data source folder."
        self.info.authors = "Ikomia team"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Dataset"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.keywords = "dataset,annotation,train,dnn"

    def create(self, param=None):
        # Create process object
        return WGISD_DatasetProcess(self.info.name, param)
