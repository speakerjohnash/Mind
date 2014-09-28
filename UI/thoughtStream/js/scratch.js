/* Take a CSV and turn it into a
thought stream visualization*/

function thoughtStream(data) {

	var words = Object.keys(data[0]),
		totals = {};

	var max = d3.max(data, function(row) { return d3.max(d3.values(row))})

	console.log(max)

	// Create Multiselect
	multiSelect(data, words)

}

/* Produces the SVG components that 
make up the thought stream */

function stream(data, selected) {

	formatted = formatData(data, selected)

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
			stream(data, selected)
		}
    });

}

$(document).ready(function() {
    
    var csvpath = "../../data/output/collective_stream.csv";
	
	d3.csv(csvpath, thoughtStream);

});