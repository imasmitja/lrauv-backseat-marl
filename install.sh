#!/bin/bash

version=$(grep -H "version" setup.cfg | cut -d '=' -f 2 | xargs)
python3 -m pip uninstall backseat_app
python3 -m pip install dist/backseat_app-$version-py3-none-any.whl
