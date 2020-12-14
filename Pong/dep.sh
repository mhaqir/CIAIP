#!/bin/bash

python3 -m pip install -U pygame==1.9.3 --user

#git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment
pip install -e . --user
cd ..
