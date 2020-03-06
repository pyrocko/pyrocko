#!/usr/bin/env python3
import os
import random
import requests


def quoteliteral(s):
    s = '\n'.join('  '+x for x in s.splitlines())
    return '''```
%s
```
''' % s


def to_message():
    message = {
        'username': 'Pyrocko Build Factory',
        'icon_url': 'https://pyrocko.org/_static/pyrocko.svg',
        'attachments': [],
    }

    text = '\n'.join(
        '%s: %s' % (k, os.environ[k]) for k in sorted(os.environ.keys()))

    env = os.environ
    keys = '''
        commit commit_message
        commit_author commit_author_name commit_author_email commit_link
        build_number build_link build_status
    '''.split()

    env = dict((k, os.environ['DRONE_' + k.upper()]) for k in keys)

    emos_success = [':%s:' % s for s in '''
        sunny star hatched_chick hamster dog butterfly sunglasses smile
        heart_eyes stuck_out_tongue smile_cat muscle +1 ok_hand clap rainbow
        beer champagne clinking_glasses medal_sports medal_military
        man_cartwheeling woman_cartwheeling fireworks'''.split()]

    emos_failure = [':%s:' % s for s in '''
        frowning_face weary skull skull_and_crossbones cold_sweat -1
        middle_finger scream_cat hankey tornado cloud_with_rain fire 8ball
        boxing_glove hocho bomb rage1'''.split()]

    env['emo'] = random.choice(
        emos_success if env['build_status'] == 'success' else emos_failure)

    artifacts = []
    for name, link in [
            ('docs', 'https://data.pyrocko.org/builds/%s/docs/'),
            ('coverage', 'https://data.pyrocko.org/builds/%s/coverage/'),
            ('wheels', 'https://data.pyrocko.org/builds/%s/wheels/')]:

        link = link % env['commit']
        r = requests.get(link)
        if r.status_code == 200:
            artifacts.append((name, link))

    env['artifacts'] = ' '.join(
        '[%s](%s)' % (name, link) for (name, link) in artifacts)

    text = '''{emo} **Build [{build_number}]({build_link}): {build_status}**
Commit: [{commit}]({commit_link}) by {commit_author} ([{commit_author_name}](mailto:{commit_author_email}))

{commit_message}

Artifacts: {artifacts}
'''.format(**env)  # noqa

    attachment = {
        'fallback': 'test',
        'color': '#33CC33' if env['build_status'] == 'success' else '#CC3333',
        'text':  text,
    }

    message['attachments'].append(attachment)
    return message


def mattermost_post(webhook, message):
    requests.post(webhook, json=message)


if __name__ == '__main__':
    webhook = os.environ.get('WEBHOOK', None)
    message = to_message()
    if webhook:
        mattermost_post(webhook, message)
    else:
        print(message)
