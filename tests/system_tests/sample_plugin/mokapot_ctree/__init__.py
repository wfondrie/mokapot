
from mokapot.model import Model
from sklearn import tree
import logging

LOGGER = logging.getLogger(__name__)

class PluginModel(Model):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("The ctree model is not production ready")
        clf = tree.DecisionTreeClassifier()
        super().__init__(clf, *args, **kwargs)

