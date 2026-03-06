import io
import base64

from .base import Scout, ScoutResult


class ResponseScout(Scout):

    format = 'webp'
    nfrequencies = 200
    show_breakpoints = True

    def update(self, context):
        from pyrocko.plot.response import plot

        buffer = io.BytesIO()

        responses = self.mantra.outlet.get_responses(
            codes=context.codes_visible,
            time=context.time)

        if not responses:
            return []

        resps = [response.get_effective() for response in responses]
        labels = [str(response.codes) for response in responses]

        styles = [
            {'alpha': 1.0 if response.codes in context.codes else 0.2}
            for response in responses]

        plot(
            resps,
            fmin=context.frequency_min or 0.01,
            fmax=context.frequency_max or 100.,
            nf=self.nfrequencies,
            labels=labels,
            styles=styles,
            filename=buffer,
            format=self.format,
            show_breakpoints=self.show_breakpoints)

        return [
            ScoutResult(
                name=self.name,
                context=context,
                image_data_base64='data:image/%s;base64,%s' % (
                    self.format,
                    base64.b64encode(buffer.getvalue()).decode('ascii')))
        ]


__all__ = [
    'ResponseScout'
]
