function thoughtStream(data) {

	// Canvas Dimensions
	var margin = {top: 20, right: 40, bottom: 30, left: 30},
    	width = document.body.clientWidth - margin.left - margin.right,
     	focusHeight = 550 - margin.top - margin.bottom,
     	contextHeight = 100;

}

$(document).ready(function() {
    
    var csvpath = "../../data/output/all_stream.csv";
	
	d3.csv(csvpath, thoughtStream);

});