#!/usr/local/bin/python3.4

"""This module starts an HTTP web service for use with processing
thoughts on Prophet for visualization and analysis purposes

Created on Jun 20, 2015
@author: Matthew Sevrens
"""

#################### USAGE ##########################

# sudo python3.3 -m mind

#####################################################

import os
import tornado.httpserver
import tornado.ioloop

from tornado_json.application import Application
from tornado.options import define, options
#from mind.sentiment_api import Sentiment_Analysis
from mind.wordstream_api import Wordstream_Analysis

# Define Some Defaults
define("port", default=443, help="run on the given port", type=int)

def main():
	"""Launches an HTTPS web service."""

	# Log that Server is Started
	print("-- Server Started --")

	# Log access to the web service
	tornado.options.parse_command_line()

	# Define valid routes
	routes = [
		#("/sentiment/?", Sentiment_Analysis),
		("/wordstream/?", Wordstream_Analysis)
	]

	# Create the tornado_json.application
	application = Application(routes=routes, settings={})

	# Create the http server
	http_server = tornado.httpserver.HTTPServer(application)

	# Start the http_server listening on default port
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
	main()