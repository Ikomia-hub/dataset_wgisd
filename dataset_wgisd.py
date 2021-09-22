from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        from dataset_wgisd.dataset_wgisd_process import DatasetWgisdFactory
        return DatasetWgisdFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        from dataset_wgisd.dataset_wgisd_widget import DatasetWgisdWidgetFactory
        return DatasetWgisdWidgetFactory()
