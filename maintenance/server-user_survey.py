#!/usr/bin/env python3
import time
import json
from urllib import parse

from http.server import BaseHTTPRequestHandler, HTTPServer

port = 58326
outfile = '/home/pyrocko/user_survey.txt'


def decode_post(data):
    system = {}
    for k, v in parse.parse_qs(data).items():
        system[k.decode()] = v[0].decode()
    return system


class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Send response status code
        content_length = int(self.headers['Content-length'])
        if content_length > 512:
            self.send_response(400, 'Bad Request')
            return

        client = self.client_address[0]
        with open(outfile, 'a') as f:
            system = decode_post(self.rfile.read(content_length))
            system['client'] = client
            system['time'] = time.time()
            f.write(json.dumps(system, indent=4))
            f.write(',\n')
            print('Metrics dumped...')

        message = b"OK"
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', len(message))
        self.end_headers()

        self.wfile.write(message)
        return


def run():
    print('Starting Pyrocko User Survey Server on http://127.0.0.1:%d...'
          % port)

    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('Running server...')
    httpd.serve_forever()


def test_post():
    import requests

    import numpy
    import scipy
    import pyrocko
    import platform
    import uuid
    import PyQt5.QtCore as qc
    data = {
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

    while True:
        print('Posting to http://127.0.0.1:%d' % port)
        try:
            requests.post('http://127.0.0.1:%d' % port, data=data)
        except Exception as e:
            print(e)
        time.sleep(5)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pyrocko User Survey Server')
    parser.add_argument(
        'action', choices=['start', 'post'],
        default='post', const='post', nargs='?',
        help='Start the server or post test data.')
    args = parser.parse_args()
    if args.action == 'start':
        run()

    if args.action == 'post':
        test_post()
