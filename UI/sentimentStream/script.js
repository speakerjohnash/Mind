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
			colorScale = d3.scale.linear().domain(polarityRange).range(["#37486F", "#457a8b"]),
			format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])}),
			xScale = d3.time.scale().domain(timeRange).range([0, 100]);

		var gradient = svg.append("defs")
		  	.append("linearGradient")
		    .attr("id", "gradient")
		    .attr("x1", "0%")
		    .attr("y1", "0%")
		    .attr("x2", "100%")
		    .attr("y2", "0%")
		    .attr("spreadMethod", "pad");

		gradient.selectAll("stop")
			.data(data)
			.enter()
			.append("stop")
			.attr("offset", function(d, i) { return xScale(format.parse(d["Post Date"])) + "%" })
			.attr("stop-color", function(d, i) { return colorScale(d["sentiment"]) })
			.attr("stop-opacity", 1);

		// Stream
		var stream = svg.append("g")
    		.attr("class", "stream");

    	var streamXScale = d3.time.scale().domain(timeRange).range([0, width]);

    	// Nest, Area and Stack
		var stack = d3.layout.stack()
			.offset("zero")
			.values(function(d) { return d.values })
			.x(function(d) { return d.date })
			.y(function(d) { return d.value });

  		var nest = d3.nest().key(function(d) { return d.key; }),
  			streamYScale = d3.scale.linear().domain(polarityRange).range([height, 0]);

  		// Area
    	var area = d3.svg.area()
		    .interpolate("basis")
		    .x(function(d) { return streamXScale(d.date); })
		    .y0(function(d) { return streamYScale(d.y0) })
			.y1(function(d) { return streamYScale(d.y0 + d.y) });

		// Format Data
		var formattedData = formatData(data),
			streamLayer = stack(nest.entries(formattedData));
			
		// Data Binding
		var streamFlow = stream.selectAll("path.layer").data(streamLayer);

		// Enter
		streamFlow.enter()
			.append("path")
			.attr("class", "layer")
			.attr("d", function(d) { return area(d.values) })
			.style("fill", "url(#gradient)");

		// Draw Axes
		var XAxis = d3.svg.axis().scale(streamXScale).orient("bottom");

  		stream.append("g")
	      .attr("class", "x axis")
	      .attr("transform", "translate(0," + height - 1 + ")")
	      .call(XAxis);

	}

})();