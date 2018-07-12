#!/usr/bin/env python3
import os
import requests

import vagrant_tests_collect


def get_webhook():
    dirname = os.path.dirname(__file__)
    filename_webhook = os.path.join(dirname, 'mattermost_webhook.link')
    if not os.path.exists(filename_webhook):
        return None

    with open(filename_webhook, 'r') as f:
        return f.read().strip()


mattermost_payload = {
    'username': 'Pyrocko',
    'icon_url': 'https://pyrocko.org/_static/pyrocko.svg',
    'text': 'Testing: %s',
}


def mattermost_post(webhook, data):
    payload = mattermost_payload.copy()
    payload['text'] = payload['text'] % data
    requests.post(webhook, json=payload)


if __name__ == '__main__':
    webhook = get_webhook()
    if webhook:
        for r in vagrant_tests_collect.iter_results():
            mattermost_post(webhook, '''
```
%s
```
''' % r)
