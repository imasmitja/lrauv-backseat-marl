# LRAUV Backseat Python Application Example

Python package that serves as an example and
[template repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
for your own LRAUV backseat application which uses the LRAUV Backseat LCM Interface python library
[here](https://bitbucket.org/mbari/lrauv-backseat-helper).

NOTE: Apps using the interface must be compatible with Python 3.8

## Quick Start

The template repository is ready to run immediately upon checkout. It will:
- Listen for backseat commands from the vehicle
- Publish a heartbeat every 5 seconds
- Handle shutdown commands

This provides a minimal working backseat application that you can extend with your own data processing.

To run the app after installation:
```bash
$ backseat_app
```

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

## Adding Custom Data Processing

The `backseat_app/main.py` file is provided as a working sample. You can:
- Use it as-is and add a processing class to handle specific data
- Use it as an example and modify it for your needs

To add custom data processing to handle specific LCM messages:

1. Create a processing module (e.g., `backseat_app/processing.py`) with handler methods
2. Import your processing class at the top of `backseat_app/main.py`
3. Instantiate your processing class in `BackseatApp.__init__()`
4. Add your handlers to the `subscribe()` dict

See the comments in `backseat_app/main.py` and the example below.

## Package Structure

This template repository implements the following structure:

```
backseat_app_package
|__ pyproject.toml
|__ setup.cfg
|__ + backseat_app
    |__ __init__.py
    |__ main.py
    |__ processing.py (you create this for custom processing)
    |__ + config
        |__ __init__.py
        |__ app_cfg.yaml
```

### Example Custom Processing Implementation

Here's a simple example of a processing module that handles sensor data:

```python
# backseat_app/processing.py
import logging
from lrauv.LCM.HandlerBase import LcmHandlerBase
from lrauv.LCM.Publisher import LcmPublisher

logger = logging.getLogger('backseat_app')

class SensorProcessor(LcmHandlerBase):
    def __init__(self, lcm_instance, cfg):
        super().__init__(lcm_instance, cfg)
        self.publisher = LcmPublisher(lcm_instance)
        self.chlorophyll_threshold = 2.5
    
    def handle_fluo(self, channel, data):
        """Process chlorophyll fluorescence data from WetLabsBB2FL sensor"""
        logger.debug(f"Handling LCM msg on channel {channel}")
        msg = self.decode(data)
        
        if 'mass_concentration_of_chlorophyll_in_sea_water' not in self.get_item_names(msg):
            return
        
        chl_var = self.get_variable('mass_concentration_of_chlorophyll_in_sea_water', msg)
        if not chl_var or len(chl_var.data) == 0:
            logger.warning("No chlorophyll data in message")
            return
        
        chl_value = chl_var.data[0]
        logger.info(f"Chlorophyll reading: {chl_value:.2f} ug/L")
        
        if chl_value > self.chlorophyll_threshold:
            logger.warning(f"High chlorophyll detected: {chl_value:.2f} ug/L")
            self.publish_alert(msg.epochMillisec, chl_value)
    
    def publish_alert(self, timestamp, value):
        """Publish a detection alert to the vehicle's slate"""
        self.publisher.clear_msg()
        self.publisher.timestamp(timestamp)
        self.publisher.add_variable(
            name='_.high_chlorophyll_detected',
            val=value,
            unit='ug/L'
        )
        self.publisher.publish('tethys_slate')
        logger.info("Published chlorophyll alert")
```

Then update `backseat_app/main.py`:

```python
# Import your processing class
from backseat_app.processing import SensorProcessor

class BackseatApp(Supervisor, LcmListener):
    def __init__(self, lcm_instance, app_cfg):
        # ... initialization code ...
        
        # Instantiate your processor
        self.sensor_processor = SensorProcessor(lcm_instance, self.cfg)
        
        # Subscribe to channels with your handlers
        self.subscribe(
            {
                self.cfg['lcm_bsd_command_channel']: self.lrauv_command_handler,
                'WetLabsBB2FL': self.sensor_processor.handle_fluo,
            }
        )
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
