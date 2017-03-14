var width = 960,
    height = 500,
    radius = 20,
    margin = 100;

var x1 = margin,
    x2 = width - margin,
    y = height / 2;
    
var drag = d3.behavior.drag()
  .origin(function(d) { return d; })
  .on("drag", dragmove);

var svg = d3.select("body").append("svg")
  .attr("width", width)
  .attr("height", height)
  .datum({
    x: width / 2,
    y: height / 2
  });

var line = svg.append("line")
  .attr("x1", x1)
  .attr("x2", x2)
  .attr("y1", y)
  .attr("y2", y)
  .style("stroke", "#f2f2f2")
  .style("stroke-linecap", "round")
  .style("stroke-width", 40);

var beginCap = svg.append("circle")
  .attr("fill", "#f1e886")
  .attr("r", radius)
  .attr("cy", y)
  .attr("cx", margin)

var endCap = svg.append("circle")
  .attr("fill", "#f1e886")
  .attr("r", radius)
  .attr("cy", function(d) { return d.y; })
  .attr("cx", function(d) { return d.x; })
  .style("cursor", "ew-resize")
  .call(drag);

function dragmove(d) {
  
  // Get the updated X location computed by the drag behavior.
  var x = d3.event.x;
  
  // Constrain x to be between x1 and x2 (the ends of the line).
  x = x < x1 ? x1 : x > x2 ? x2 : x;
  
  // This assignment is necessary for multiple drag gestures.
  // It makes the drag.origin function yield the correct value.
  d.x = x;
  
  // Update the circle location.
  endCap.attr("cx", x);
}