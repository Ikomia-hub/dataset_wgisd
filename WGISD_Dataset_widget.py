from ikomia import utils, core, dataprocess
import WGISD_Dataset_process as processMod
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class WGISD_DatasetWidget(core.CProtocolTaskWidget):

    def __init__(self, param, parent):
        core.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.WGISD_DatasetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.browse_folder = utils.append_browse_file(grid_layout=self.grid_layout, label="Dataset folder",
                                                     path=self.parameters.data_folder_path,
                                                     mode=QFileDialog.Directory)

        self.browse_class_file = utils.append_browse_file(grid_layout=self.grid_layout, label="Classes file",
                                                         path=self.parameters.class_file_path,
                                                         mode=QFileDialog.ExistingFile)

        self.check_mask = utils.append_check(grid_layout=self.grid_layout, label="Load segmentation masks",
                                            checked=self.parameters.load_mask)

        # PyQt -> Qt wrapping
        layout_ptr = utils.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onApply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.data_folder_path = self.browse_folder.path
        self.parameters.class_file_path = self.browse_class_file.path
        self.parameters.load_mask = self.check_mask.isChecked()

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class WGISD_DatasetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "WGISD_Dataset"

    def create(self, param):
        # Create widget object
        return WGISD_DatasetWidget(param, None)
