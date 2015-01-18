(function() {

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
	          		"key" : selected[i], 
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

	/* Prepare data after loading from csv 
	and draw necessary UI elements */

	function makeStream(data) {

		// Canvas Dimensions
		var focusMargin = {top: 20, right: 40, bottom: 30, left: 30},
			contextMargin = {top: 430, right: 10, bottom: 20, left: 40},
	    	width = document.body.clientWidth - focusMargin.left - focusMargin.right,
	     	focusHeight = 500 - focusMargin.top - focusMargin.bottom,
	     	contextHeight = 500 - contextMargin.top - contextMargin.bottom;

	    // Data
		var words = Object.keys(data[0]),
			totals = calculateTotals(data, words);

		// Top Words
		var sorted = sortObjectIntoArray(totals),
			topWords = [],
			sLen = sorted.length;

		for (var i=0; i<150; i++) {
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
			contextScale = d3.time.scale().domain(timeRange).range([0, width]);

		// Selection and Brushing
		var brush = d3.svg.brush()
    		.x(contextScale)
    		.on("brush", brushed);

    	multiSelect(data, topWords, true)

    	// Setting Up Canvas
    	var svg = d3.select(".chart").append("svg")
			.attr("width", width + focusMargin.left + focusMargin.right)
			.attr("height", focusHeight + focusMargin.top + focusMargin.bottom)
			.append("g")
			.attr("transform", "translate(" + focusMargin.left + "," + focusMargin.top + ")");

		// Containers for Focus and Context
		var focus = svg.append("g")
    		.attr("class", "focus")
    		.attr("transform", "translate(" + focusMargin.left + "," + focusMargin.top + ")");

		var context = svg.append("g")
    		.attr("class", "context")
    		.attr("transform", "translate(" + contextMargin.left + "," + contextMargin.top + ")");

		// Axes
		var focusXAxis = d3.svg.axis().scale(focusScale).orient("bottom"),
    		contextXAxis = d3.svg.axis().scale(contextScale).orient("bottom");

    	// Draw Context
    	stream(contextData, ["Total"])

    	// Draw a Stream
    	function stream(data, wordList) {

    	}

    	// Set Domain of Focus
    	function brushed() {
 			focusScale.domain(brush.empty() ? contextScale.domain() : brush.extent());
			// Redraw Focus and Focus Axis
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
					stream(data, selected)
				}
		    });

		}

	}
    
    var csvpath = "../../data/output/all_stream.csv";
	
	d3.csv(csvpath, makeStream);

})();