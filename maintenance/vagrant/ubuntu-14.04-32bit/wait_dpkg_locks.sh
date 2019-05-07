#!/bin/bash

sleep 5

while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1 ; do
    echo -en "Waiting for other software managers to finish..."
    sleep 5
done

while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
    echo -en "Waiting for other software managers to finish..."
    sleep 5
done
