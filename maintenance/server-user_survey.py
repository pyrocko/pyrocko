#!/usr/bin/env python3
import time
import json
import requests
import GeoIP

from http.server import BaseHTTPRequestHandler, HTTPServer

port = 58326
outfile = '/tmp/user_survey.txt'

mattermost_webhook = ''

gi = GeoIP.new(GeoIP.GEOIP_MMAP_CACHE)

mattermost_payload = {
    'username': 'Pyrocko',
    'icon_url': 'https://pyrocko.org/_static/pyrocko.svg',
    'text': 'Yay, someone from **%s** contributed to the survey! :pray:',
}


def mattermost_post(data):
    payload = mattermost_payload.copy()
    payload['text'] = payload['text'] % data['country']
    requests.post(mattermost_webhook, json=payload)


class PyrockoSurveyHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Send response status code
        content_length = int(self.headers['Content-length'])
        if content_length > 512:
            self.send_response(400, 'Bad Request')
            return

        with open(outfile, 'a') as f:
            data = json.loads(self.rfile.read(content_length).decode())
            data['client'] = self.headers.get('X-Real-IP', '')
            data['country'] = gi.country_name_by_addr(data['client'])
            data['country_code'] = gi.country_code_by_addr(data['client'])
            data['time'] = time.time()
            f.write(json.dumps(data, indent=4))
            f.write(',\n')
            print('Metrics dumped...')

        message = b"OK"
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', len(message))
        self.end_headers()

        self.wfile.write(message)

        mattermost_post(data)
        return


def run():
    print('Starting Pyrocko User Survey Server on http://127.0.0.1:%d...'
          % port)

    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, PyrockoSurveyHandler)
    print('Running server...')
    httpd.serve_forever()


def _get_data():
    import numpy
    import scipy
    import pyrocko
    import platform
    import uuid
    import PyQt5.QtCore as qc

    return {
        'node-uuid': uuid.getnode(),
        'platform.architecture': platform.architecture(),
        'platform.system': platform.system(),
        'platform.release': platform.release(),
        'python': platform.python_version(),
        'pyrocko': pyrocko.__version__,
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'qt': qc.PYQT_VERSION_STR,
    }


def test_post():
    addr = 'http://localhost:%d' % port
    data = _get_data()

    while True:
        print('Posting to %s' % addr)
        try:
            requests.post(addr, json=data)
        except Exception as e:
            print(e)
        time.sleep(5)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pyrocko User Survey Server')
    parser.add_argument(
        'action', choices=['start', 'post', 'webhook'],
        help='Start the server or post test data to it. '
             'Webhook integrates to mattermost')
    args = parser.parse_args()

    if args.action == 'start':
        run()

    if args.action == 'post':
        test_post()

    if args.action == 'webhook':
        mattermost_post()
