/* Take a CSV and turn it into a
thought stream visualization*/

function thoughtStream(data) {

	// Canvas
	var margin = {top: 20, right: 40, bottom: 30, left: 30},
    	width = document.body.clientWidth - margin.left - margin.right,
     	height = 350 - margin.top - margin.bottom;

	// Data
	var words = Object.keys(data[0]),
		totals = calculateTotals(data, words);

	// Top Words
	var sorted = sortObject(totals),
		topWords = []; 

	for (var i=0; i<50; i++) {
		topWords.push(sorted[i]["key"])
	}

	// Create Multiselect
	multiSelect(data, topWords, true)

	// Create SVG
	var svg = d3.select(".chart").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	// Select Top Words
	$("#multiselect").multiselect('select', topWords.slice(0, 5));

	// Stream
	stream(data, topWords.slice(0, 5))

}

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
of a word over a give time period */

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

/* Produces the SVG components that 
make up the thought stream */

function stream(data, selected) {

	// Return if nothing selected
	if (selected == null) {
		d3.selectAll("path").data(function() {return []}).exit().remove()
		return
	}

	// SVG
	var svg = d3.select(".chart svg");

	// Canvas
	var margin = {top: 20, right: 40, bottom: 30, left: 30},
    	width = document.body.clientWidth - margin.left - margin.right,
     	height = 350 - margin.top - margin.bottom,
     	max = d3.max(data, function(row) { return d3.max(d3.values(row))});

	// Stack
	var stack = d3.layout.stack()
		.offset("wiggle")
		.values(function(d) { return d.values; })
		.x(function(d) { return d.date; })
		.y(function(d) { return d.value; });

	// Nest
  	var nest = d3.nest().key(function(d) { return d.key; });

	// Format Data
	var formatted = formatData(data, selected),
		layers = stack(nest.entries(formatted));

	// Scales
	var timeRange = d3.extent(formatted, function(d) { return d.date; }),
		color = d3.scale.linear().domain([0, selected.length]).range(["#457a8b", "#455a8b"]),
		x = d3.time.scale().domain(timeRange).range([0, width]),
      	y = d3.scale.linear().range([height-10, 0]);

      	console.log(timeRange)
      	console.log(d3.max(formatted, function(d) { return d.y0 + d.y; }))

    // Change Scale
    //y.domain([0, d3.max(formatted, function(d) { return d.y0 + d.y; })]);

    // Same Scale
    y.domain([0, 150]);

    // Area
    var area = d3.svg.area()
	    .interpolate("cardinal")
	    .x(function(d) { return x(d.date); })
	    .y0(function(d) { return y(d.y0); })
	    .y1(function(d) { return y(d.y0 + d.y); });

	// Draw Stream
	var flows = svg.selectAll("path").data(layers)

	// Enter
	flows.enter()
		.append("path")
		.attr("class", "layer")
		.attr("d", function(d) { return area(d.values); })
		.style("fill", function(d, i) { return color(i); });

	// Exit
	flows.exit().remove();

    // Transition
    flows.transition()
      .duration(1000)
	  .attr("d", function(d) { return area(d.values); })

	// Hover
	svg.selectAll("path")
      .attr("opacity", 1)
      .on("mouseover", function(d, i) {
        svg.selectAll(".layer").transition()
        .duration(250)
        .attr("opacity", function(d, j) {
          return j != i ? 0.6 : 1;
    })})

}

/* Prepare a set of words for
visualization */

function formatData(data, selected) {

    var formatted = [],
    	format = d3.time.format("%m/%d/%Y"),
        row = "";

	data.forEach(function(day) {
		for (var i=0, l=selected.length; i<l; i++) {
			row = {
				"value" : parseInt(day[selected[i]], 10), 
          		"key" : selected[i], 
          		"date" : format.parse(day["Post Date"])
			}
			formatted.push(row)
		}
	})

	return formatted

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
		onChange : function (element, checked) {
			selected = $("select.multiselect").val()
			stream(data, selected)
		}
    });

}

$(document).ready(function() {
    
    var csvpath = "../../data/output/processed_stream.csv";
	
	d3.csv(csvpath, thoughtStream);

});