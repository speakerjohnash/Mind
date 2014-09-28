/* Take a CSV and turn it into a
thought stream visualization*/

function thoughtStream(data) {

	// Canvas
	var margin = {top: 20, right: 40, bottom: 30, left: 30},
    	width = document.body.clientWidth - margin.left - margin.right,
     	height = 350 - margin.top - margin.bottom;

	// Data
	var words = Object.keys(data[0]),
		totals = {},
		max = d3.max(data, function(row) { return d3.max(d3.values(row))});

	// D3
	var tools = {
		"width" : width,
		"height" : height
	}

	// Create Multiselect
	multiSelect(data, words, tools)

	// Create SVG
	var svg = d3.select(".chart").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

}

/* Produces the SVG components that 
make up the thought stream */

function stream(data, selected, tools) {

	// SVG
	var svg = d3.select(".chart svg"),
		width = tools["width"],
		height = tools["height"];

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

	// Draw Stream


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

function multiSelect(data, words, tools) {

	// Create Multiselect
	var widget = d3.select("body").append("div").classed("widget", true),
		select = widget.append("select").classed("multiselect", true),
		params = {
			"multiple" : "multiple",
	 		"data-placeholder" : "Add thoughts"
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
		includeSelectAllOption : true,
		maxHeight : 250,
		onChange : function (element, checked) {
			selected = $("select.multiselect").val()
			stream(data, selected, tools)
		}
    });

}

$(document).ready(function() {
    
    var csvpath = "../../data/output/collective_stream.csv";
	
	d3.csv(csvpath, thoughtStream);

});