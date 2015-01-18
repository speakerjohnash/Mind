$(document).ready(function() {

	/* Sort an Object */

	function sortObject(obj) {

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
	of all words */

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
		var sorted = sortObject(totals),
			topWords = [],
			sLen = sorted.length;

	}
    
    var csvpath = "../../data/output/all_stream.csv";
	
	d3.csv(csvpath, makeStream);

});