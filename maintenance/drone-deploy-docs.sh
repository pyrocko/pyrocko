#!/bin/bash


set -e

[ ! -z "$DRONE" ] || exit 1


keypath=/tmp/deploy-www-key-$$

umask 0077
echo "$WWW_KEY" > $keypath
umask 0022

ssh -o StrictHostKeyChecking=no -i $keypath "${WWW_USER}@${WWW_HOST}" \
    "${DRONE_COMMIT}" "pyrocko"

rm $keypath
