(function timeSelect() {
  var width = 960,
      height = 80,
      timeSpaceHeight = 60,
      xSteps = d3.range(10, width, 10),
      ySteps = d3.range(0),
      brushStart,
      dateBegin,
      dateEnd;

  var now = new Date,
      year = now.getFullYear(),
      then = (new Date).setFullYear(year + 100),
      timeFisheye = d3.fisheye.scale(d3.time.scale).domain([now, then]).range([0, width]).focus(0);

  var xFisheye = d3.fisheye.scale(d3.scale.identity).domain([0, width]).focus(0);

  var linearTimeScale = d3.time.scale().domain([0, width]).range([now, then]);

  var svg = d3.select("#chart").append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", "translate(-.5,-.5)");

  svg.append("rect")
      .attr("class", "background")
      .attr("width", width)
      .attr("height", timeSpaceHeight)

  var xLine = svg.selectAll(".x")
      .data(xSteps)
      .enter().append("line")
      .attr("class", "x")
      .attr("y2", timeSpaceHeight);

  var yLine = svg.selectAll(".y")
      .data(ySteps)
      .enter().append("line")
      .attr("class", "y")
      .attr("x2", width);

  var timeLine = d3.svg.axis().scale(timeFisheye).ticks(10).orient("bottom")

  // Brush
  var brush = d3.svg.brush().x(xFisheye);

  brush.on("brushstart", function(){
    var xPos = d3.mouse(this)[0]
    brushStart = xPos
    dateBegin = linearTimeScale(brushStart)
  })

  brush.on("brushend", function(){
    dateEnd = linearTimeScale(d3.mouse(this)[0])
    if (dateEnd < dateBegin) {
      var tempDate = dateEnd
      dateEnd = dateBegin
      dateBegin = tempDate
    }
  })

  brush.on("brush", function(){

    var newPixel = timeFisheye(linearTimeScale(brushStart)),
        xMouse = d3.mouse(this)[0],
        x,
        width;

    if (xMouse > brushStart) {
      x = newPixel
      width = xMouse - newPixel
    } else {
      x = xMouse
      width = newPixel - xMouse
    }
    
    gBrush.selectAll(".extent")
        .attr("x", x)
        .attr("width", width)

    brush.extent([x, x + width])
    
  })

  var gBrush = svg.append("g")
    .attr("class", "brush")
    .call(brush);

  gBrush.selectAll("rect")
    .attr("height", timeSpaceHeight);

  svg.on("mousemove", function() {
    var mouse = d3.mouse(this);
    xFisheye.focus(mouse[0]);
    timeFisheye.focus(mouse[0]);
    redraw();
  });

  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0, " + timeSpaceHeight + ")")
    .call(timeLine);

  redraw();

  function redraw() {

    if (dateBegin && dateEnd) {
      var newbStart = timeFisheye(dateBegin),
          newbEnd = timeFisheye(dateEnd);

      gBrush.selectAll(".extent")
        .attr("x", newbStart)
        .attr("width", newbEnd - newbStart)
    }

    svg.select(".x.axis").call(timeLine);
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);
  }

})();