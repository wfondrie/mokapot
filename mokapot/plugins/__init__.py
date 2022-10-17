
import importlib


def load_plugin(plugin_name):
    pkg = importlib.import_module(f"mokapot.plugins.{plugin_name}")
    return pkg


if __name__ == "__main__":
    pkg = load_plugin("errorplugin")
    pkg.PluginModel()
    # load_plugins("errorplugin")
