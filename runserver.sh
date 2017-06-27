#!/usr/bin/env bash

# activates the debugger, activates the automatic reloader
export FLASK_DEBUG=1

export FLASK_APP=server.py
flask run --host=0.0.0.0
