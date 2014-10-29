(function chart() {
  var width = 960,
      height = 60,
      xSteps = d3.range(10, width, 20),
      ySteps = d3.range(0);

  var timeDomains = [
      new Date,
      d3.time.day.offset(new Date, 1),
      d3.time.week.offset(new Date, 1),
      d3.time.month.offset(new Date, 1),
      d3.time.year.offset(new Date, 1),
      d3.time.year.offset(new Date, 10),
      d3.time.year.offset(new Date, 100)
  ]

  var timeRanges = [0, 160, 320, 480, 640, 800, 960]

  var now = new Date,
      year = now.getFullYear(),
      then = (new Date).setFullYear(year + 100),
      timeFisheye = d3.fisheye.scale(d3.time.scale).domain([now, then]).range([0, width]).focus(0),
      timeIntervalScale = d3.fisheye.scale(d3.time.scale).domain(timeDomains).range(timeRanges);

  var xFisheye = d3.fisheye.scale(d3.scale.identity).domain([0, width]).focus(0);

  var svg = d3.select("#chart").append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", "translate(-.5,-.5)");

  svg.append("rect")
      .attr("class", "background")
      .attr("width", width)
      .attr("height", 45);

  var xLine = svg.selectAll(".x")
      .data(xSteps)
      .enter().append("line")
      .attr("class", "x")
      .attr("y2", 45);

  var yLine = svg.selectAll(".y")
      .data(ySteps)
      .enter().append("line")
      .attr("class", "y")
      .attr("x2", width);

  var timeLine = d3.svg.axis().scale(timeFisheye).ticks(10).orient("bottom")

  // Brush
  var brush = d3.svg.brush().x(xFisheye);

  var gBrush = svg.append("g")
    .attr("class", "brush")
    .call(brush);

  gBrush.selectAll("rect")
    .attr("height", 45);

  svg.on("mousemove", function() {
    var mouse = d3.mouse(this);
    xFisheye.focus(mouse[0]);
    timeFisheye.focus(mouse[0]);
    redraw();
  });

  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0, 44)")
    .call(timeLine);

  redraw();

  function redraw() {
    svg.select(".x.axis").call(timeLine);
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);
  }

})();

/*
(function chart4() {

  // Various accessors that specify the four dimensions of data to visualize.
  function x(d) { return d.income; }
  function y(d) { return d.lifeExpectancy; }
  function radius(d) { return d.population; }
  function color(d) { return d.region; }

  // Chart dimensions.
  var margin = {top: 5.5, right: 19.5, bottom: 12.5, left: 39.5},
      width = 960,
      height = 500 - margin.top - margin.bottom;

  // Various scales and distortions.
  var xScale = d3.fisheye.scale(d3.scale.log).domain([300, 1e5]).range([0, width]),
      yScale = d3.fisheye.scale(d3.scale.linear).domain([20, 90]).range([height, 0]),
      radiusScale = d3.scale.sqrt().domain([0, 5e8]).range([0, 40]),
      colorScale = d3.scale.category10().domain(["Sub-Saharan Africa", "South Asia", "Middle East & North Africa", "America", "Europe & Central Asia", "East Asia & Pacific"]);

  // The x & y axes.
  var xAxis = d3.svg.axis().orient("bottom").scale(xScale).tickFormat(d3.format(",d")).tickSize(-height),
      yAxis = d3.svg.axis().scale(yScale).orient("left").tickSize(-width);

  // Create the SVG container and set the origin.
  var svg = d3.select("#chart4").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  // Add a background rect for mousemove.
  svg.append("rect")
      .attr("class", "background")
      .attr("width", width)
      .attr("height", height);

  // Add the x-axis.
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  // Add the y-axis.
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis);

  // Add a y-axis label.
  svg.append("text")
      .attr("class", "y label")
      .attr("text-anchor", "end")
      .attr("x", -6)
      .attr("y", 6)
      .attr("dy", ".75em")
      .attr("transform", "rotate(-90)")
      .text("Years");

  // Add a dot per nation. Initialize the data at 1800, and set the colors.
  var dot = svg.append("g")
      .attr("class", "dots")
      .selectAll(".dot")
      .data([])
      .enter().append("circle")
      .attr("class", "dot")
      .style("fill", function(d) { return colorScale(color(d)); })
      .call(position)
      .sort(function(a, b) { return radius(b) - radius(a); });

  // Add a title.
  dot.append("title")
      .text(function(d) { return d.name; });

  // Positions the dots based on data.
  function position(dot) {
    dot .attr("cx", function(d) { return xScale(x(d)); })
        .attr("cy", function(d) { return yScale(y(d)); })
        .attr("r", function(d) { return radiusScale(radius(d)); });
  }

  svg.on("mousemove", function() {
    var mouse = d3.mouse(this);
    xScale.distortion(2.5).focus(mouse[0]);
    yScale.distortion(2.5).focus(mouse[1]);

    dot.call(position);
    svg.select(".x.axis").call(xAxis);
    svg.select(".y.axis").call(yAxis);
  });
  
})(); */