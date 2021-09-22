from ikomia import utils, core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from dataset_wgisd.dataset_wgisd_process import DatasetWgisdParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class DatasetWgisdWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = DatasetWgisdParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.browse_folder = pyqtutils.append_browse_file(grid_layout=self.grid_layout, label="Dataset folder",
                                                     path=self.parameters.data_folder_path,
                                                     mode=QFileDialog.Directory)

        self.browse_class_file = pyqtutils.append_browse_file(grid_layout=self.grid_layout, label="Classes file",
                                                         path=self.parameters.class_file_path,
                                                         mode=QFileDialog.ExistingFile)

        self.combo_seg_mask_mode = pyqtutils.append_combo(grid_layout=self.grid_layout, label="Segmentation masks")
        self.combo_seg_mask_mode.addItem("None")
        self.combo_seg_mask_mode.addItem("Instance")
        self.combo_seg_mask_mode.addItem("Panoptic")
        self.combo_seg_mask_mode.setCurrentText(self.parameters.seg_mask_mode)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.data_folder_path = self.browse_folder.path
        self.parameters.class_file_path = self.browse_class_file.path
        self.parameters.seg_mask_mode = self.combo_seg_mask_mode.currentText()

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class DatasetWgisdWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "dataset_wgisd"

    def create(self, param):
        # Create widget object
        return DatasetWgisdWidget(param, None)
