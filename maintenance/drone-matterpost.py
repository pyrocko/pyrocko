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

    env = dict(
        (k, os.environ.get(
            'DRONE_%s' % k.upper(), 'DRONE_' + k.upper() + ' undefined'))
        for k in keys)

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

    artifacts = {}
    for name, link in [
            ('docs', 'https://data.pyrocko.org/builds/%s/docs/'),
            ('coverage', 'https://data.pyrocko.org/builds/%s/coverage/index.html'),  # noqa
            ('wheels', 'https://data.pyrocko.org/builds/%s/wheels/'),
            ('deb', 'https://data.pyrocko.org/builds/%s/deb/'),
            ('apps', 'https://data.pyrocko.org/builds/%s/apps/')]:

        link = link % env['commit']
        r = requests.get(link)
        if r.status_code == 200:
            artifacts[name] = link

    env['artifacts'] = ' '.join(
        '[%s](%s)' % (name, link) for (name, link) in artifacts.items())

    env['commit_message'] = '\n'.join(env['commit_message'].splitlines()[:1])

    total_coverage = None
    if 'coverage' in artifacts:
        r = requests.get(os.path.join(
            os.path.dirname(artifacts['coverage']), 'status.json'))

        coverage = r.json()

        statements = 0
        missing = 0
        excluded = 0

        for file in coverage['files'].values():
            statements += file['index']['nums'][1]
            excluded += file['index']['nums'][1]
            missing += file['index']['nums'][3]

        if statements > 0:
            total_coverage = (statements - missing) / statements

    text = '''{emo} **Build [{build_number}]({build_link}): {build_status}**
[{commit:.7s}]({commit_link}): {commit_message} - [{commit_author_name}](mailto:{commit_author_email})

Artifacts: {artifacts}
'''.format(**env)  # noqa

    if total_coverage:
        text += 'Total coverage: %.1f%%\n' % (total_coverage * 100.)

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
