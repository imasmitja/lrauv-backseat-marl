# LRAUV Backseat Python Application Example

Python package that serves as an example and
[template repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
for your own LRAUV backseat application which uses the LRAUV Backseat LCM Interface python library
[here](https://bitbucket.org/mbari/lrauv-backseat-helper).

NOTE: Apps using the interface must be compatible with Python 3.8

## Install App Package

Scripts have been included for conveniently installing the python app package.

### Build

Build a wheel with:
```bash
$ ./build.sh
```

### Install

Install the wheel with:
```bash
$ ./install.sh
```

### Cleanup

Remove build files with:
```bash
$ ./cleanup.sh
```

### Do All Three

Perform cleanup/build/install:
```bash
$ ./cycle.sh
```

## Use Interface In Your Own Package

This template repository implements the structure and example code outlined below.

You may use the lrauv backseat LCM interface as shown in
the example code below. This would be your entry_point
console_script located in your app package as `main.py`
structured like so:

```
your_app_package
|__ pyproject.toml
|__ setup.cfg
|__ + your_app
    |__ __init__.py
    |__ your_processing.py
    |__ main.py
    |__ + config
        |__ __init__.py
        |__ app_cfg.yaml
```

### Example Code Implementation

As stated above, this would be in the `main.py` for your
package which would implement the LRAUV backseat LCM interface.

```python
try:
    import importlib.resources as ilr
    ilr.files
except AttributeError:
    import importlib_resources as ilr

import logging

import lcm

from lrauv.LCM.Listener import LcmListener
from lrauv.supervisor.Supervisor import Supervisor

from lrauv.supervisor import Logger
logger = None


# TODO change this to use your code
from backseat_app.processing import ProcessingClass


class BackseatApp(Supervisor, LcmListener):

    def __init__(self, lcm_instance, app_cfg):
        Supervisor.__init__(self, lcm_instance, app_cfg)
        LcmListener.__init__(self, lcm_instance)
        global logger
        logger, _ = Logger.configure_logger(name=app_cfg['app_name'])

        # TODO change this to use your code
        self.processing_class = ProcessingClass(lcm_instance, self.cfg)

        # request LCM data from LRAUV
        self.request_slate(self.cfg['lrauv_data'])

        # subscribe LCM handlers
        self.set_timeout(timeout_sec=2.5)
        # TODO add your handlers for data requested in config file
        self.subscribe(
            {
                self.cfg['lcm_bsd_command_channel']: self.lrauv_command_handler,
                'WetLabsUBAT': self.your_processing_class.handle_ubat,
                'WetLabsBB2FL': self.your_processing_class.handle_fluo
            }
        )


    def spin(self):
        """
        Main loop. Handle subscriber callbacks until instructed to shutdown.

        :return: 1 upon exit
        """
        while self.run:
            try:
                # listen to incoming LCM messages with a timeout
                self.listen()
                self.strobe_heartbeat(heart_rate_sec=5.0)
                self.request_data.request()
            except KeyboardInterrupt:
                logger.error("KeyboardInterrupt: aborting.")
                break
        # power down
        self.unsubscribe()
        self.shutdown()
        logger.info("Terminating.")
        exit(1)


def main():
    import argparse
    from lrauv.config.AppConfig import read_config

    default_cfg = ilr.files('backseat_app.config').joinpath('app_config.yaml')

    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Runs the backseat app.')
    parser.add_argument("-c", "--config", default=default_cfg,
                        type=str, help="set path to app config file")

    args = parser.parse_args()

    # read in app configuration
    cfg = read_config(args.config)

    # init LCM object
    lc = lcm.LCM(cfg['lcm_url'])
    # init BackseatApp
    backseat_app = BackseatApp(lc, cfg)
    # run
    backseat_app.spin()


if __name__ == "__main__":
    main()

```

### Config

You must also add a config module in your app as described in the example app structure above. It must
contain a modified yaml file like the one in `backseat_app/config/app_cfg.yaml`

## Install Your App As A Service

Along with the Backseat Interface, a helper script is also installed to allow you to create and install
your app as a Linux service.

```bash
$ lrauv_install_service <your app name> <top level of your app package> <entry_point console_script executable for your app (installed to ~/.local/bin)>
```

This will install your app as a service with the name your_app.service, and you can interact with
it using `systemctl`.

#### Status
```bash
$ sudo systemctl status your_app.service
```

#### Stop
```bash
$ sudo systemctl stop your_app.service
```

#### Restart
```bash
$ sudo systemctl restart your_app.service
```

#### Disable

This will disable your app from running automatically when the backseat boots up.

```bash
$ sudo systemctl status your_app.service
```

#### Enable

This will enable your app to automatically run when the backseat boots up.

```bash
$ sudo systemctl status your_app.service
```
