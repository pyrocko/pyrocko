#!/bin/bash


set -e

[ ! -z "$DRONE" ] || exit 1


keypath=/tmp/rsync-key-$$

umask 0077
echo "$RSYNC_KEY" > $keypath
umask 0022

temppath=/tmp/rsync-temp-$$

mkdir -p $temppath/$2

rsync -av $1 $temppath/$2
rsync -rv -e "ssh -o StrictHostKeyChecking=no -i $keypath" $temppath/ ${RSYNC_USER}@${RSYNC_HOST}:

rm $keypath
