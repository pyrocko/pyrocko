# pvc - Pyrocko's easy installer

Dead simple installation of Pyrocko for people who want to try it out:

```sh
source <(curl -fsSL https://pyrocko.org/pvc_init) && pvc install
```

This installs the latest release of Pyrocko from PyPI (with GUI
dependencies) into a virtual environment in `~/.pyrocko/venv`. Run `pvc`
without arguments for help.

- `pvc_init`: shell part (bash and zsh), defines the `pvc` shell function. On
  install, a copy is put into `~/.pyrocko` and sourced from the user's
  existing `~/.bashrc` and `~/.zshrc`.
- `pvc_install`: Python part, creates the virtual environment and runs pip.

Both files are served from https://pyrocko.org/ (`pvc_init`, `pvc_install`)
and need to be updated there after changes.
