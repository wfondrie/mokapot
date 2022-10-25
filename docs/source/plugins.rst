
Writting plugins for **mokapot**
================================

This is a tutorial on how to write plugins for mokapot. 

The plugin architecture allows you to extend mokapot without
requiring any changes to the core of mokapot.

Right now it allows you to modify the data before
it gets used by the models or implement a new model.

Plugins are python packages that export a `BasePlugin` object
to an entry point called `mokapot.plugins`. It requires setting an
entry point up in your package configuration (We will walk you though it together).
This entails adding a specification in your package configuration that lets your build
system know that one of your classes should be "installed" inside mokapot, once installed
there, mokapot will be able to find it and use it.

If using `setuptools <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_ with `setup.py` you can do that like this:

    setup(
        ...
        entry_points={
            'mokapot.plugins': [
                'myplugin = my_package:MyPluginClass',
            ],
        },
    )

If using `pyproject.toml` you can do that like this:

    [project.entry-points."mokapot.plugins"]
    myplugin = "my_package:MyPluginClass"

If using `poetry <https://python-poetry.org/docs/pyproject/#plugins>`_ you will need these lines in your `pyproject.toml`:

    [tool.poetry.plugins."mokapot.plugins"]
    "myplugin" = "my_package:MyPluginClass"

In these examples "myplugin" is the name you wish to give to your plugin and
"my_package:MyPluginClass" is the name of the class you wish to export.

An example plugin can be found in the tests for mokapot in
`tests/system_tests/sample_plugin`

The plugin needs to implement the `BasePlugin` class.
You do not need to implement all the methods, but you can if you want to.


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

Using your plugin
-----------------

Once mokapot and your plugin are installed in the same
python environeent you can use your plugin by requesting it
on the command line:

    mokapot .... --plugin myplugin ...

In the case of the example plugin, called `mokapot_ctree` you
can use it like this:

    mokapot .... --plugin mokapot_ctree ...

Note that when calling mokapot with the `--help` option, the
arguments for the plugin will show up in the bottom as a new
section.

    mokapot --help
    ...
    mokapot_ctree:
      --yell               Whether to yell into the command line while executing
