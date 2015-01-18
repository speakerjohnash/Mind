(function() {

	/* Sort an Object */

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
		var margin = {top: 20, right: 40, bottom: 30, left: 30},
	    	width = document.body.clientWidth - margin.left - margin.right,
	     	focusHeight = 550 - margin.top - margin.bottom,
	     	contextHeight = 100;

	    // Data
		var words = Object.keys(data[0]),
			totals = calculateTotals(data, words);

		// Top Words
		var sorted = sortObjectIntoArray(totals),
			topWords = [],
			sLen = sorted.length;

		// Generate Context Data Stream
		var contextData = calculateDayTotals(data, words);

		// Scales and Conversions
		var format = d3.time.format("%m/%d/%Y"),
			timeRange = d3.extent(data, function(d) { return format.parse(d["Post Date"])}),
			x = d3.time.scale().domain(timeRange).range([0, width]);

		// Selection and Brushing
		var brush = d3.svg.brush()
    		.x(x)
    		.on("brush", brushed);

    	multiSelect(data, topWords, true)

    	// Setting Up Canvas
    	var svg = d3.select(".chart").append("svg")
			.attr("width", width + margin.left + margin.right)
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    	// Set Domain of Focus
    	function brushed() {
 			
		}

	}
    
    var csvpath = "../../data/output/all_stream.csv";
	
	d3.csv(csvpath, makeStream);

})();