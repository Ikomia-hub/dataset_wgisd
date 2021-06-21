from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class WGISD_Dataset(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        from WGISD_Dataset.WGISD_Dataset_process import WGISD_DatasetProcessFactory
        return WGISD_DatasetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        from WGISD_Dataset.WGISD_Dataset_widget import WGISD_DatasetWidgetFactory
        return WGISD_DatasetWidgetFactory()
