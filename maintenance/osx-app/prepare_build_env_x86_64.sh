#!/bin/bash

export HOMEBREW_CELLAR=/usr/local/Cellar
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/usr/local/bin/brew shellenv)"
arch -x86_64 brew install python@3.11 pyqt@5
