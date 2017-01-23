$(document).ready(function() {

	/* Sort an Object's keys and values
	into an array */

	function sortObjectIntoArray(obj) {

	    var arr = [];

	    for (var prop in obj) {
	        if (obj.hasOwnProperty(prop)) {
	            arr.push({
	                'key': prop,
	                'value': obj[prop]
	            });
	        }
	    }

	    arr.sort(function(a, b) { return b.value - a.value; });

	    return arr;
	}


	/* Prepare data for visualization */

	function formatData(data, wordList) {

	    var formatted = [],
	    	format = d3.time.format("%m/%d/%Y"),
	        row = "";

		data.forEach(function(day) {
			for (var i=0, l=wordList.length; i<l; i++) {
				row = {
					"value" : parseInt(day[wordList[i]], 10), 
	          		"key" : wordList[i], 
	          		"date" : format.parse(day["Post Date"])
				}
				formatted.push(row)
			}
		})

		return formatted

	}

	/* Calculate the total usage
	of each word */

	function calculateTotals(data, words) {

		var totals = {}

		for (var k=0; k<words.length; k++) {
			totals[words[k]] = 0
		}

		for (var i=0; i<data.length; i++) {
			for (var ii=0; ii<words.length; ii++) {
				var word = words[ii]
				if (word != "Post Date") {
					totals[word] += parseInt(data[i][word], 10)
				}
			}
		}

		return totals

	}

	/* Calculate the total usage
	of all words per day */

	function calculateDayTotals(data, words) {

		var totals = []

		for (var i=0, l=data.length; i<l; i++) {
			var date = data[i]["Post Date"]
			dayTotal = 0
			for (var ii=0, k=words.length; ii<k; ii++) {
				dayTotal += parseInt(data[i][words[ii]])
			}
			totals.push({"Post Date": date, "Total": dayTotal})
		}

		return totals

	}

	/* Fill in any missing days with zero
	values */

	function fillInDays(data) {
		makeStream(data)
	}

	/* Prepare data after loading from csv 
	and draw necessary UI elements */

	function makeStream(data) {

		// Canvas Dimensions
		var focusMargin = {top: 120, right: 20, bottom: 20, left: 20},
			contextMargin = {top: 20, right: 20, bottom: 20, left: 20},
	    	width = document.body.clientWidth - focusMargin.left - focusMargin.right,
	    	axisHeight = 20,
	    	numWords = 5,
	     	focusHeight = 430,
	     	contextHeight = 80,
	     	svgHeight = focusHeight + contextHeight + (2 * axisHeight),
	     	wordsInSelect = 5;

	    // Data
		var words = Object.keys(data[0]),
			totals = calculateTotals(data, words);

		// Top Words
		var sorted = sortObjectIntoArray(totals),
			topWords = [],
			sLen = sorted.length;

		if (sLen < wordsInSelect) wordsInSelect = sLen

		for (var i=0; i<wordsInSelect; i++) {
			if (sorted[i]["key"] != "Post Date") {
				topWords.push(sorted[i]["key"])
			}
		}

		// Generate Context Data Stream
		var contextData = calculateDayTotals(data, words);

		// Scales and Conversions
		var format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])}),
			focusScale = d3.time.scale().domain(timeRange).range([0, width]),
			contextScale = d3.time.scale().domain(timeRange).range([0, width])
			contextYScale = d3.scale.linear().range([0, contextHeight]),
			focusYScale = d3.scale.linear().range([0, focusHeight]),
			type_colors = {
        		"Predict": "#5A4E8C",
        		"Ask": "#5CB85C",
        		"Thought": "#C4C4C4",
        		"State": "#428BCA",
        		"Reflect": "#A33333"
    		};

		// Selection and Brushing
		var brush = d3.svg.brush()
    		.x(contextScale)
    		.on("brush", brushed);

    	multiSelect(data, topWords, true)
		$("#multiselect").multiselect('select', topWords.slice(0, numWords));

    	// Setting Up Canvas
    	var svg = d3.select(".chart").append("svg")
			.attr("width", width + focusMargin.left + focusMargin.right)
			.attr("height", svgHeight + 1)
			.append("g")
			.attr("transform", "translate(" + focusMargin.left + ", 0)");

		// Containers for Focus and Context
		var context = svg.append("g")
    		.attr("class", "context");

		var focus = svg.append("g")
    		.attr("class", "focus");

		// Axes
		var focusXAxis = d3.svg.axis().scale(focusScale).orient("top"),
    		contextXAxis = d3.svg.axis().scale(contextScale).orient("top");

    	// Nest, Area and Stack
		var stack = d3.layout.stack()
			.offset("silhouette")
			.values(function(d) { return d.values; })
			.x(function(d) { return d.date; })
			.y(function(d) { return d.value; });

		var contextArea = d3.svg.area()
		    .interpolate("basis")
		    .x(function(d) { return contextScale(d.date); });

		var focusArea = d3.svg.area()
		    .interpolate("basis")
		    .x(function(d) { return focusScale(d.date); });

  		var nest = d3.nest().key(function(d) { return d.key; });

    	// Draw Streams
    	streams(topWords.slice(0, numWords))

    	// Add Brush
		context.append("g")
	    	.attr("class", "x brush")
	    	.call(brush)
			.selectAll("rect")
  			.attr("height", contextHeight);

  		// Draw Axes
  		focus.append("g")
	      .attr("class", "x axis")
	      .attr("transform", "translate(0," + svgHeight + ")")
	      .call(focusXAxis);

	    context.append("g")
	      .attr("class", "x axis")
	      .attr("transform", "translate(0," + (contextHeight + axisHeight) + ")")
	      .call(contextXAxis);

    	// Draw Streams
    	function streams(wordList) {

    		var lata = data;

    		// Return if no words selected
			if (wordList == null) {
				focus.selectAll("path").data(function() {return []}).exit().remove();
				lata = contextData
				wordList = ["Total"]
			}

			// Format Data
			var formattedContext = formatData(contextData, ["Total"])
				contextLayer = stack(nest.entries(formattedContext)),
				formattedFocus = formatData(lata, wordList),
				focusLayers = stack(nest.entries(formattedFocus));

			console.log(formattedFocus)

			// Scale Adjustments
			var color = d3.scale.linear().domain([0, wordList.length]).range(["#457a8b", "#455a8b"]);

    		contextYScale.domain([0, d3.max(contextData, function(d) { return d["Total"]; })]);
    		focusYScale.domain([0, d3.max(formattedFocus, function(d) { return d.y0 + d.y; })]);

			// Area
    		contextArea
			    .y0(function(d) { return contextYScale(d.y0); })
			    .y1(function(d) { return contextYScale(d.y0 + d.y); });

			focusArea
				.y0(function(d) { return focusYScale(d.y0); })
			    .y1(function(d) { return focusYScale(d.y0 + d.y); });

			// Data Binding
			var contextFlow = context.selectAll("path.layer").data(contextLayer),
				focusFlows = focus.selectAll("path.layer").data(focusLayers);

			// Enter
			contextFlow.enter()
				.append("path")
				.attr("class", "layer")
				.attr("d", function(d) { return contextArea(d.values); })
				.style("fill", function(d, i) { return color(i); });

			focusFlows.enter()
				.append("path")
				.attr("class", "layer")
				.attr("d", function(d) { return focusArea(d.values); })
    			.attr("transform", "translate(0, " + (contextHeight + axisHeight) + ")")
				.style("fill", function(d, i) { return type_colors[d.key]; });

			// Exit
			focusFlows.exit().remove();

			// Transition
			focusFlows.transition()
				.duration(1000)
				.attr("d", function(d) { return focusArea(d.values); })
				.style("fill", function(d, i) { return type_colors[d.key]; });

    	}

    	// Set Domain of Focus
    	function brushed() {
 			focusScale.domain(brush.empty() ? contextScale.domain() : brush.extent());
 			focus.selectAll("path.layer").attr("d", function(d) { return focusArea(d.values); })
 			focus.select(".x.axis").call(focusXAxis);
		}

		/* Creates a multiple selection widget
		for selection of ideas to display */

		function multiSelect(data, words) {

			// Create Multiselect
			var widget = d3.select("body").append("div").classed("widget", true),
				select = widget.append("select").classed("multiselect", true),
				params = {
					"multiple" : "multiple",
			 		"data-placeholder" : "Add thoughts",
			 		"id" : "multiselect"
				}

			// Create Basic Markup
			select.attr(params)
				.selectAll("option")
				.data(words)
				.enter()
				.append("option")
				.text(function(d){return d});

			// Construct Widget
			$('.multiselect').multiselect({
				enableFiltering : true,
				maxHeight : 250,
				includeSelectAllOption: true,
				onChange : function (element, checked) {
					selected = $("select.multiselect").val()
					streams(selected)
				}
		    });

		}

	}
    
    var csvpath = "../../data/output/perspective.csv";
	
	d3.csv(csvpath, fillInDays);

});