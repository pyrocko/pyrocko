#!/bin/bash
docker build nest -t pyrocko-nest
docker build docs -t pyrocko-docs
docker build aux -t pyrocko-aux
docker build pyrocko -t pyrocko
