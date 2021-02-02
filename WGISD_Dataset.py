from ikomia import dataprocess
import WGISD_Dataset_process as processMod
import WGISD_Dataset_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class WGISD_Dataset(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.WGISD_DatasetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.WGISD_DatasetWidgetFactory()
