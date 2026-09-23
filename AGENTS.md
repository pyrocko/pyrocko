# AGENTS.md

Guidance for AI coding agents working in the Pyrocko repository. This
complements, and does not replace, `CONTRIBUTING.md` (development workflow,
CI, branching, code style, release process) and `README.md` (installation,
project overview). Read those first for anything not covered here.

## What Pyrocko is

Pyrocko is a seismology toolkit for Python: waveform I/O and processing,
station/event metadata, Green's function stores (`pyrocko.gf`), synthetic
seismogram modelling, geodesy/geometry, and GUI tools (Snuffler, Sparrow).
Most of the package is pure Python with a handful of C extensions
(`src/ext/*.c`, built via `setup.py`/`pyproject.toml`).

## Repository layout

- `src/` — the `pyrocko` package (installed as top-level modules, not a
  `pyrocko/` subdirectory here; e.g. `src/trace.py` becomes
  `pyrocko.trace`).
  - `src/squirrel/` — Squirrel, the data management/access layer used
    throughout the newer parts of the codebase (waveform, station, event,
    response data). `squirrel/tool/commands/` holds the `squirrel <cmd>`
    CLI subcommands. `squirrel/service/` is a Tornado-based web service
    with a Vue 3 frontend under `squirrel/service/page/`.
  - `src/gato/` — Generalized Array Toolkit, for seismic array processing
    (beamforming/CSM, delay-and-sum grids, array response). Has its own
    CLI (`gato`) and Qt GUI (`gato/gui/`).
  - `src/gf/` — Green's function store handling and synthetic modelling.
  - `src/gui/` — Snuffler (waveform viewer/picker) and Sparrow (3D globe
    viewer).
  - `src/guts.py`, `src/guts_array.py` — Pyrocko's declarative
    serialization framework (YAML/XML), used pervasively for config
    objects, model classes, and on-disk formats. Classes typically end
    with `.T()` attribute declarations and a `guts_prefix` module
    variable that namespaces the YAML tag.
  - `src/ext/` — C extensions.
- `test/` — pytest suite, split into `test/base`, `test/gf`, `test/gui`,
  `test/examples` (the four groups also used by CI); `test/data` holds
  test fixtures.
- `doc/` — Sphinx documentation sources; build with `make html` in `doc/`.
- `webapps/` — build tooling for the vendored web frontends (see below).
- `maintenance/` — release/packaging scripts, not part of the library.

## Building and testing

- Installing: see `README.md`. For source-tree development, the usual
  approach is `pip install -e .` in a virtual environment (editable
  install), or `python install.py system`.
- **Check whether your environment's `pyrocko` install is editable
  before trusting that edits under `src/` take effect.** A non-editable
  install (`pip install .` without `-e`, or a plain `pip install
  pyrocko`) copies `src/` into `site-packages` at install time; further
  edits to the working tree are invisible to anything that imports
  `pyrocko` (including the test suite) until the package is reinstalled
  or the site-packages copy is synced. Verify with:
  ```sh
  python -c "import pyrocko, os; print(os.path.realpath(pyrocko.__file__))"
  ```
  If this does not point into the repository's `src/`, either reinstall
  editable or reinstall after each change before relying on test results.
- Run the test suite with `python -m pytest` (or target a subset, e.g.
  `python -m pytest test/base/test_squirrel.py`). Tests need network
  access and/or example data for some cases; those are typically guarded
  by decorators (see `test/common.py`, `require_internet`-style helpers).
- Lint with `flake8` before committing; a `pre-commit` hook is provided
  (`pre-commit install`). CI enforces flake8, and code must follow PEP8
  plus the additional conventions in `CONTRIBUTING.md` (British English,
  `i`/`n` index/count naming, message capitalization rules, etc.).

## Experimental subsystems

Some modules are explicitly marked experimental and call
`pyrocko.util.experimental_feature_used('pyrocko.<name>')` on import or
first use, which emits a one-time `UserWarning`. As of this writing this
includes `pyrocko.carpet`, `pyrocko.gato`, and the Squirrel service
(`pyrocko.squirrel.service`). Treat their APIs as more likely to change
than the rest of the library, and don't be surprised by the warning in
test output — it is expected, not a bug.

## Squirrel service frontend

`src/squirrel/service/page/` contains a Vue 3 + Quasar + vue-flow
frontend. Its third-party JS/CSS/font assets under `vendor/` are
committed directly into the repository rather than fetched by a build
step — this is a deliberate choice (those libraries change rarely, and
baking them in avoids adding a build dependency to the Python package).
Don't propose replacing this with a build-time fetch; if the vendored
assets need updating, `webapps/squirrel-service/scripts/sync-assets.js`
is the tool for that.

## Squirrel/Gato "mantra" processing pipelines

`pyrocko.squirrel.mantra.Mantra` describes a connected graph of
`pyrocko.squirrel.operators` (or `pyrocko.gato.operators`) that
transforms/derives data on top of a `Squirrel`. Mantra configs are
Guts-serializable YAML, loaded via `--mantra PATH[:PATTERN,...]` in
several CLI tools (`squirrel mantra`, `squirrel service`). When adding
code that touches this area, keep in mind that mantra/operator names
become part of on-disk YAML tags and derived data codes (NSLCE `extra`
field conventions) — renaming things is a compatibility-affecting change.

## Commit and branch conventions

See `CONTRIBUTING.md` for the full policy; the essentials:

- Commit subject: lower-case start, `component: ` prefix, imperative
  mood, e.g. `squirrel: add merge_codes functionality`.
- Branch names: `feature/<name>`, `bugfix/<name>`, `hotfix/<name>`,
  `docs/<name>`, `ci/<name>`.
- `main` is the stable branch; feature work happens on topic branches and
  is rebased before merging.
- Notable changes belong in `CHANGELOG.md` (Keep a Changelog format).
  Keep it brief: not every commit needs an entry, only ones that matter
  to a user of Pyrocko. Write entries in plain, non-technical language,
  about one line each, distinct in tone from commit messages. Security
  issues and breaking changes are the exception and may get a longer,
  more detailed entry.

## License

GPLv3-or-later. Don't introduce dependencies or vendored code under
incompatible licenses without flagging it explicitly.
