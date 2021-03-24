# Pyrocko development environment and contribution guide

## Language

Pyrocko is written in the Python programming language (versions ==2.7 and >=3.4).

## Deployment

Pyrocko uses Setuptools for its installation script. See `setup.py` and
`setup.cfg` in the project root directory.

## Testing

Nosetests is used for testing. To run all tests, run

```sh
python -m nose test
```

## CI

Drone CI tests are run on any commits pushed to the repository. By default,
flake8 is run, tests are run parallel in 4 groups, test coverage is measured,
and docs are built. Success/failure is reported to the Pyrocko Hive.

Pushing to specific branches triggers extra pipelines:

* `hptime`: tests are run in both time handling modes (double test time)
* `pip`: pip wheels and sdist are built, tested and uploaded to
  PyPi-testing
* `conda`: anaconda packages are built and tested
* `deploy-docs`: docs are published
* `candidate`: same as pip + anaconda + hptime
* `release`: same as pip + anaconda + deploy-docs, packages are additionally
  published and package tests are skipped

## Versioning and releases

Git is used for version control. Use development branches for new features.
Master branch should always point to a stable version.

The Pyrocko project  adheres to a [calendar dating versioning scheme](https://calver.org/#scheme), using the vYYYY.MM.DD format.

Notable changes must be documented in the file `CHANGELOG.md`. The format of
the change log is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/).

### Commit message conventions

* start with lower case
* colon-prepend affected component
* try to use imperative form
* examples:
  - `docs: add section about weighting`
  - `waveform targets: correct typo in component names`
  - `waveform targets: fix issues with misaligned traces`

### Branching policy

* Use topic branches to develop new features. Follow the naming convention:
  - `feature/<name>`:
    Any code changes for a new module or use case should be done on a
    feature branch. This branch is created based on the current `master` branch.
  - `bugfix/<name>`:
    Any necessary fixes after that should be done on the bugfix branch.
  - `hotfix/<name>`:
    If there is a need to fix a blocker, do a temporary patch, apply a critical
    framework or configuration change that should be handled immediately,
    it should be created as a Hotfix.
  - `ci/<name>`:
    For branches working on the CI pipelines.
  - `docs/<name>`:
    For branches concerning changes and additions only to the docs.
  - `merge/<name>`:
    A temporary branch for resolving merge conflicts, usually between the
    latest development and a feature or Hotfix branch.

* Open a pull request and use the Gitea-tags `Need Review`, `Need Revision`, to
  signal its state.
* When a topic is complete, all tests pass and it is rebased to current master:
  merge with `--ff-only` and don't forget to update the changelog.
* The `master` branch should always point to a stable version.
* Extra CI pipelines and pipeline steps are run on branches named `release`,
  `candidate`, `pip`, `conda`, `deploy-docs`, and `hptime`. See also CI
  section above.

### Rebase small changes before pushing

Try to rebase little changes on top of master (or any other development branch)
before pushing, it makes the history much better readable. Here is a safe way
to do so.

*If we have already commited and merged changes to local master:*

```sh
git checkout master
git fetch origin    # important, otherwise we rebase to outdated
git rebase origin/master
git push origin master
```

with `git config --global pull.rebase true` this can be shortcutted to

```sh
git pull
```

*Or after we have commited to a feature branch:*

```sh
git checkout feature
git fetch origin
git rebase origin/master
git checkout master
git merge origin/master --ff-only
git merge feature --ff-only
git push origin master
```

If during push it refuses to upload ('not fast forward...') then repeat the
procedure, because someone else has pushed between your fetch and push.

**Tip:** use `rebase -i ...` to simplify/fixup/beautify your changeset.

## Code style

Pyrocko source code must follow the PEP8 coding standards. It must pass the
code style check provided by the `flake8` tool.

Additionally,

* use i/n convention for indices and counts
  - e.g. `for istation in range(nstations):`
* do not abbreviate words unless this would result in ridiculously long names
* use British english, e.g.
  - 'modelling' rather than 'modeling'
  - 'analyser' rather than 'analyzer'
  - 'optimiser' rather than 'optimizer'
* log and exception messages:
  - capital beginning
  - final period
  - Progress actions should end with `...`, e.g. `Generating report's archive...`
  - e.g. `raise ProblemDataNotAvailable('No problem data available (%s).' % dirname)`
  - in-text names must be quoted; not needed after colons
* docstrings:
  - Docs are built with Sphinx, use rst syntax.
  - Follow the usual convention 1 line summary, blank line, description.

## Documentation

Pyrocko's documentation is built using the `Sphinx` tool. See the `docs`
in the project root directory. Build with `make html` in `docs`.

*Text style rules:*

* titles: only capitalize first word
* use British english

## License

GNU General Public License, Version 3, 29 June 2007

Copyright © 2018 Helmholtz Centre Potsdam GFZ German Research Centre for
Geosciences, Potsdam, Germany and University of Kiel, Kiel, Germany.

Pyrocko is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. Pyrocko is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

