
from pyrocko import response_plot

resps, labels = response_plot.load_response_information(
    'test_response.resp', 'resp')

response_plot.plot(
    responses=resps, labels=labels, filename='test_response.png',
    fmin=0.001, fmax=400., dpi=75.)
