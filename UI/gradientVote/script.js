var width = 960;
var height = 500;
var radius = 20;
var margin = 100;

var x1 = margin;
var x2 = width - margin;
var y = height / 2;
    
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
  .style("stroke", "black")
  .style("stroke-linecap", "round")
  .style("stroke-width", 5);

var beginCap = svg.append("circle")
  .attr("r", radius)
  .attr("cy", y)
  .attr("cx", margin)

var circle = svg.append("circle")
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
  circle.attr("cx", x);
}