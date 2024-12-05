
Writing plugins for **mokapot**
================================

This is a tutorial on how to write plugins for mokapot.
The plugin architecture allows you to write Python packages that extend mokapot without requiring any changes to mokapot itself.
Currently, it allows you to modify the data before it gets used by the models or implement a new model.

Plugins are python packages that export a ``BasePlugin`` object to an entry point called ``mokapot.plugins``.
It requires setting an entry point up in your package configuration, which we will walk you though together.
This entails adding a specification in your package configuration that lets your build system know that one of your classes should be "installed" inside mokapot; once installed, mokapot will be able to find your plugin and use it.

Configuration for you plugin
----------------------------

If using `setuptools <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_ with ``setup.py`` you can do that like this:::

    setup(
        ...
        entry_points={
            'mokapot.plugins': [
                'myplugin = my_package:MyPluginClass',
            ],
        },
    )

If using setuptools with ``pyproject.toml`` you can do that like this:::

    [project.entry-points."mokapot.plugins"]
    myplugin = "my_package:MyPluginClass"

If using `poetry <https://python-poetry.org/docs/pyproject/#plugins>`_ you will need these lines in your ``pyproject.toml``:::

    [tool.poetry.plugins."mokapot.plugins"]
    "myplugin" = "my_package:MyPluginClass"

In these examples ``myplugin`` is the name you wish to give to your plugin and ``my_package:MyPluginClass`` is the name of the class you wish to export.

Writing your plugin
-------------------

An example plugin can be found in the `tests for mokapot <https://github.com/wfondrie/mokapot/tree/master/tests/system_tests/sample_plugin>`_.
Each plugin must implement the ``BasePlugin`` class.
You do not need to implement all of the methods, but you can if you want to.::

    class Plugin(BasePlugin):
        def add_arguments(parser: _ArgumentGroup) -> None:
            # Define any arguments your plugin may need
            parser.add_argument(
                "--yell", action="store_true", help="Whether to yell into the command line while executing"
            )

        def get_model(self, config) -> mokapot.model.Model:
            # Define any models, they should be a subclass of
            # mokapot.model.Model
            if config.yell:
                LOGGER.warning("I LOVE WRITTING PLUGINS")
            return PluginModel(
                train_fdr=config.train_fdr,
                max_iter=config.max_iter,
                direction=config.direction,
                override=config.override,
                subset_max_train=config.subset_max_train,
            )

        def process_data(self, data, config):
            # Modify the data before it gets used by the model
            LOGGER.info("Processing data within the plugin")
            return data

These methods control the following:
- The ``add_arguments()`` method defines new command line arguments that will be used by your plugin.
- The ``get_model()`` method creates a new mokapot model that will be used.
- The ``process_data()`` method adds data processing steps that are executed before model training.

Using your plugin
-----------------

Once mokapot and your plugin are installed in the same Python environment you can use your plugin by requesting it on the command line:::

    mokapot ... --plugin myplugin ...

In the case of the example plugin, called `mokapot_ctree` you can use it like this:::

    mokapot ... --plugin mokapot_ctree ...

Note that when calling mokapot with the `--help` option, the arguments for the plugin will show up in the bottom as a new section:::

    mokapot --help
    ...
    mokapot_ctree:
      --yell               Whether to yell into the command line while executing
