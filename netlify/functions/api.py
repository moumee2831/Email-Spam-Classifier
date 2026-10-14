import os
import sys

# Ensure root directory is in Python path for importing app
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import serverless_wsgi
from app import app

def handler(event, context):
    """
    AWS Lambda / Netlify Function handler wrapping Flask via serverless-wsgi.
    """
    return serverless_wsgi.handle_request(app, event, context)
