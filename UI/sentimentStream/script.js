(function sentimentStream() {

	var csvpath = "../../data/output/sentiment_stream.csv";
	
	d3.csv(csvpath, buildStream);

	/* Prepare data for visualization */

	function formatData(data) {

	    var formatted = [],
	    	format = d3.time.format("%m/%d/%Y"),
	        row = "";

		data.forEach(function(day) {
			row = {
				"value" : parseFloat(day["sentiment"], 10), 
          		"key" : "sentiment", 
          		"date" : format.parse(day["Post Date"])
			}
			formatted.push(row)
		})

		return formatted

	}

	function buildStream(data) {

		var width = 800,
    		height = 200;

		var svg = d3.select(".canvas-frame").append("svg")
		    .attr("width", width)
		    .attr("height", height);

		// Gradients
		var polarityRange = d3.extent(data, function(d) { return d["sentiment"] }),
			format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])});

		// Stream
		var field = svg.append("g")
    		.attr("class", "field");

    	var x = d3.time.scale().domain(timeRange).range([0, width]),
    		y = d3.scale.linear().domain([-1,1]).range([height, 0]);

    	// Line 
    	var line = d3.svg.line()
    		.interpolate("basis")
    		.x(function(d) { return x(d.date); })
    		.y(function(d) { return y(d.value); });
		
		// Draw Axes
		var XAxis = d3.svg.axis().scale(x).orient("bottom");

  		field.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(0, " + (height - 20) + ")")
			.call(XAxis);

		var formattedData = formatData(data)

		svg.append("line")
    		.style("stroke", "black")  // colour the line
			.attr("x1", 0)     // x position of the first end of the line
			.attr("y1", height / 2)      // y position of the first end of the line
			.attr("x2", width)     // x position of the second end of the line
			.attr("y2", height / 2);    // y position of the second end of the line

		svg.append("g")
			.append("path")
			.datum(formattedData)
			.attr("class", "line")
			.attr("d", line);

	}

})();