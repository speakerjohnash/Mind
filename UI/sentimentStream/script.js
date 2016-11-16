(function sentimentStream() {

	var csvpath = "../../data/output/sentiment_stream.csv";
	
	d3.csv(csvpath, buildStream);

	/* Prepare data for visualization */

	function formatData(data) {

	    var positivity = [],
	    	mood = [],
	    	format = d3.time.format("%m/%d/%Y"),
	        row = "";

		data.forEach(function(day) {
			p_row = {
				"value" : parseFloat(day["vote"], 10), 
				"key" : "positivity", 
				"date" : format.parse(day["Post Date"])
			}

			positivity.push(p_row)

			if (parseFloat(day["mood"], 10) != 0) {
				m_row = {
					"value" : parseFloat(day["mood"], 10), 
					"key" : "mood", 
					"date" : format.parse(day["Post Date"])
				}
				mood.push(m_row)
			}
				 
		})

		return [positivity, mood]

	}

	function buildStream(data) {

		var width = 800,
    		height = 200;

		var svg = d3.select(".canvas-frame").append("svg")
		    .attr("width", width)
		    .attr("height", height + 20);

		// Data
		var moodAndPositivity = formatData(data),
			positivity = moodAndPositivity[0],
			mood = moodAndPositivity[1];

		// Gradients
		var positivityRange = d3.extent(positivity, function(d) { return d["value"] }),
			moodRange = d3.extent(mood, function(d) { return d["value"] }),
			format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])});

    	var x = d3.time.scale().domain(timeRange).range([0, width]),
    		y = d3.scale.linear().domain([-1, 1]).range([height, 0])
    		mY = d3.scale.linear().domain(moodRange).range([height, 0]);

    	// Line 
    	var line = d3.svg.line()
    		.interpolate("bundle")
    		.x(function(d) { return x(d.date); })
    		.y(function(d) { return mY(d.value); });

    	// TODO: Add a vertical gradient with blue for sad, yellow for happy and maybe green for neutral

    	// TODO: Load prophet thoughts via brush selection

    	// TODO: Scatterplot
		svg.selectAll("dot")
			.data(mood)
			.enter().append("circle")
			.attr("r", 2)
			.attr("cx", function(d) { return x(d.date); })
			.attr("cy", function(d) { return mY(d.value); });
		
		// Draw Axes
		var XAxis = d3.svg.axis().scale(x).orient("bottom");

		var field = svg.append("g").attr("height", height);
		
  		var axis = svg.append("g")
			.attr("class", "x axis")
			.attr("transform", "translate(0, " + height + ")")
			.call(XAxis);

		field.append("path")
			.datum(mood)
			.attr("class", "line")
			.attr("d", line);

		field.append("line")
    		.style("stroke", "black")  // colour the line
			.attr("x1", 0)     // x position of the first end of the line
			.attr("y1", height / 2)      // y position of the first end of the line
			.attr("x2", width)     // x position of the second end of the line
			.attr("y2", height / 2);    // y position of the second end of the line


	}

})();