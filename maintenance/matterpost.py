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


def quoteliteral(s):
    s = '\n'.join('  '+x for x in s.splitlines())
    return '''```
%s
```
''' % s


def to_message(results):
    message = {
        'username': 'Pyrocko Testing Department',
        'icon_url': 'https://pyrocko.org/_static/pyrocko.svg',
        'attachments': [],
    }

    lines = []
    lines2 = []
    n_ok = 0
    n = 0
    truncated = False
    for result in results:
        if result.result:
            res = result.result.split()[0]
        else:
            res = ''

        emos = {
            'OK': ':champagne:',
            'FAILED': ':-1:',
            'ERROR': ':bomb:'}

        lines.append(
            '{emo} {r.package} {branch} {r.box} {py_version}: '
            '{r.result}'.format(
                emo=emos.get(res, ':poultry_leg:'),
                r=result,
                branch=result.branch or 'x',
                py_version=(
                    'py%s' % result.py_version if result.py_version else 'x')))

        result.log = None
        result.skips = []
        result.prerequisite_versions = {}
        if result.result and result.result.startswith('OK'):
            n_ok += 1
        n += 1

        if sum(len(x) for x in lines2) < 3000:
            lines2.append(result.dump())
        elif not truncated:
            lines2.append('... [message truncated]')
            truncated = True

    all_ok = n_ok == n

    if all_ok:
        summary = '**Well done, all test suites are running smoothly!**'
    elif n_ok == 0:
        summary = '**All test suites failed! ' \
            'All programmers must return to their work-stations, immediatly!**'
    else:
        summary = '**%i out of %i test suites failed.**' % (
            n - n_ok, n)

    attachment = {
        'fallback': 'test',
        'color': '#33CC33' if all_ok else '#CC3333',
        'text': summary + '\n\n\n\n' + quoteliteral('\n\n'.join(lines2)),
        'fields': [{'short': True, 'value': line} for line in lines],
    }

    message['attachments'].append(attachment)

    return message


def mattermost_post(webhook, message):
    requests.post(webhook, json=message)


if __name__ == '__main__':
    webhook = get_webhook()
    if webhook:
        message = to_message(vagrant_tests_collect.iter_results())
        mattermost_post(webhook, message)
