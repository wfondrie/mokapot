import logging
from argparse import _ArgumentGroup

from sklearn import tree

from mokapot.model import Model
from mokapot.plugins import BasePlugin

LOGGER = logging.getLogger(__name__)


class PluginModel(Model):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("The ctree model is not production ready")
        clf = tree.DecisionTreeClassifier()
        super().__init__(clf, *args, **kwargs)


class Plugin(BasePlugin):
    def add_arguments(parser: _ArgumentGroup) -> None:
        parser.add_argument(
            "--yell", action="store_true", help="Yell at the user"
        )

    def get_model(self, config):
        if config.yell:
            LOGGER.warning("Yelling at the user")
        return PluginModel(
            train_fdr=config.train_fdr,
            max_iter=config.max_iter,
            direction=config.direction,
            override=config.override,
            subset_max_train=config.subset_max_train,
        )

    def process_data(self, data, config):
        LOGGER.info("Processing data within the plugin")
        return data
