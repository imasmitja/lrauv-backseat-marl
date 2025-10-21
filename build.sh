#!/bin/bash

version=$(grep -H "version" setup.cfg | cut -d '=' -f 2 | xargs)
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BACKSEAT_APP=$version python3 -m build --wheel
