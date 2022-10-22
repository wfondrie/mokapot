
from mokapot.model import Model
from sklearn import tree


class PluginModel(Model):
    def __init__(self, *args, **kwargs):
        clf = tree.DecisionTreeClassifier()
        super().__init__(clf, *args, **kwargs)

