
import dash
from flask import Flask

server = Flask(__name__)
app = dash.Dash(name = __name__, server = server)
app.config.suppress_callback_exceptions = True