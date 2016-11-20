(function sentimentStream() {

	var csvpath = "../../data/output/sentiment_stream.csv",
		globalData = {};
	
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

	movingAvg = function(n) {
		return function (points) {
			points = points.map(function(each, index, array) {
				var to = index + n - 1;
				var subSeq, sum;
				if (to < points.length) {
					subSeq = array.slice(index, to + 1);
					sum = subSeq.reduce(function(a,b) { return [a[0] + b[0], a[1] + b[1]]; });
					return sum.map(function(each) { return each / n; });
				}
				return undefined;
			});
			points = points.filter(function(each) { return typeof each !== 'undefined' });
			pathDesc = d3.svg.line().interpolate("basis")(points)
			return pathDesc.slice(1, pathDesc.length);
		}
	}

	function buildStream(data) {

		globalData = data;

		var width = window.innerWidth,
			height = window.innerHeight / 2.5,
			legendWidth = 60,
			axisHeight = 20;

		var svg = d3.select(".canvas-frame").append("svg")
			.attr("width", width - legendWidth)
			.attr("height", height + axisHeight);

		var legend = d3.select(".canvas-frame").append("div")
			.attr("class", "legend")
			.style("height", height + "px")
			.style("width", 35 + "px");

		legend.append("span")
			.attr("class", "face")
			.style("margin-top", height / 2 + "px")

		// Data
		var moodAndPositivity = formatData(data),
			positivity = moodAndPositivity[0],
			mood = moodAndPositivity[1];

		// Gradients
		var positivityRange = d3.extent(positivity, function(d) { return d["value"] }),
			moodRange = d3.extent(mood, function(d) { return d["value"] }),
			format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])}),
			colorScale = d3.scale.linear().domain([-1, 0, 1]).range(['#694a69', 'steelblue', 'yellow']);

		var x = d3.time.scale().domain(timeRange).range([0, width]),
			y = d3.scale.linear().domain([-1, 1]).range([height, 0])
			mY = d3.scale.linear().domain(moodRange).range([height, 0]);

		// Line 
		var line = d3.svg.line()
			.interpolate(movingAvg(4))
			.x(function(d) { return x(d.date); })
			.y(function(d) { return mY(d.value); });

		// TODO: Load prophet thoughts via brush selection

		// Scatterplot
		svg.selectAll("dot")
			.data(mood)
			.enter().append("circle")
			.attr("r", 1.5)
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
			.attr("class", "fat-line")
			.attr("d", line);

		var path = field.append("path")
			.datum(mood)
			.attr("class", "line")
			.attr("d", line);

		var pathEl = path.node();
		var pathLength = pathEl.getTotalLength();

		var cheat = d3.scale.linear().domain([height, 0]).range([-1, 1]);

		var circle = 
        svg.append("circle")
          .attr("cx", 100)
          .attr("cy", 350)
          .attr("r", 3)
          .attr("fill", "red");

		// Interactive
		svg.on("mousemove", function() {
			var mouse = d3.mouse(this),
				date = x.invert(mouse[0]),
				beginning = mouse[0], 
				end = pathLength;
				
			var target;

			while (true) {
				target = Math.floor((beginning + end) / 2);
				pos = pathEl.getPointAtLength(target);
				if ((target === end || target === beginning) && pos.x !== x) {
					break;
				}
				if (pos.x > x) end = target;
				else if (pos.x < x) beginning = target;
				else                break; //position found
			}

			circle
		        .attr("opacity", 1)
		        .attr("cx", mouse[0])
		        .attr("cy", pos.y);

			var color = colorScale(cheat(pos.y))

			d3.select(".legend span").style("background", color)

			// TODO: Get y value at x-position
			// Cast that number to a color and set the color of the face to the gradient
  		});

		field.append("line")
			.style("stroke", "black")  // colour the line
			.attr("x1", 0)	 // x position of the first end of the line
			.attr("y1", height / 2)	  // y position of the first end of the line
			.attr("x2", width - legendWidth)	 // x position of the second end of the line
			.attr("y2", height / 2);	// y position of the second end of the line

		// Gradient
		svg.append("linearGradient")
			.attr("id", "temperature-gradient")
			.attr("gradientUnits", "userSpaceOnUse")
			.attr("x1", 0).attr("y1", y(-1))
			.attr("x2", 0).attr("y2", y(1))
			.selectAll("stop")
			.data([
				{offset: "0%", color: "#694a69"}, // black, red, purple
				{offset: "50%", color: "steelblue"},
				{offset: "100%", color: "yellow"}
			])
			.enter().append("stop")
			.attr("offset", function(d) { return d.offset; })
			.attr("stop-color", function(d) { return d.color; });

	}

})();