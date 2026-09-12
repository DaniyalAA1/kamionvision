"""KamionVision — appraise a used semi-tractor from photos alone.

Four stages, deliberately separate so each one can be inspected on its own:

    gate      reject or re-ask before any money is spent on inference
    evidence  one structured vision call, every claim bound to a photo_id
    pricing   a hedonic regression over harvested comparables + an interval
    report    the card a buyer reads, with the trace behind it

`pipeline.appraise` wires them together and returns an `Appraisal` that
serialises to JSON whole - the CLI and the web app are both thin renderers
over that one object.
"""

__version__ = "1.0.0"
