(function chart() {
  var width = 960,
      height = 380,
      xSteps = d3.range(10, width, 16),
      ySteps = d3.range(0);

  var xFisheye = d3.fisheye.scale(d3.scale.identity).domain([0, width]).focus(360),
      yFisheye = d3.fisheye.scale(d3.scale.identity).domain([0, height]).focus(90);

  var svg = d3.select("#chart").append("svg")
      .attr("width", width)
      .attr("height", height)
    .append("g")
      .attr("transform", "translate(-.5,-.5)");

  svg.append("rect")
      .attr("class", "background")
      .attr("width", width)
      .attr("height", height);

  var xLine = svg.selectAll(".x")
      .data(xSteps)
    .enter().append("line")
      .attr("class", "x")
      .attr("y2", height);

  var yLine = svg.selectAll(".y")
      .data(ySteps)
    .enter().append("line")
      .attr("class", "y")
      .attr("x2", width);

  redraw();

  svg.on("mousemove", function() {
    var mouse = d3.mouse(this);
    xFisheye.focus(mouse[0]);
    yFisheye.focus(mouse[1]);
    redraw();
  });

  function redraw() {
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);
    yLine.attr("y1", yFisheye).attr("y2", yFisheye);
  }
})();