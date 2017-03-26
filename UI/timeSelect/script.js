(function timeSelect() {
  var width = window.innerWidth / 1.25,
      height = 80,
      timeSpaceHeight = 60,
      xSteps = d3.range(10, width, 10),
      ySteps = d3.range(0),
      brushStart,
      dateBegin,
      dateEnd;

  var t = d3.transition()
    .duration(750);

  var now = new Date,
      year = now.getFullYear(),
      then = (new Date).setFullYear(year + 1),
      timeFisheye = d3.fisheye.scale(d3.time.scale).domain([now, then]).range([0, width]).focus(0),
      timeFormat = d3.time.format("%m/%d/%Y");

  var xFisheye = d3.fisheye.scale(d3.scale.identity).domain([0, width]).focus(0);

  var chartContainer = d3.select("#chart")
    .style("width", width + 40 + "px");

  var linearTimeScale = d3.time.scale().domain([0, width]).range([now, then]);

  var svg = chartContainer.append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g");

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

  // Draw Axes
  var formatMillisecond = d3.time.format(".%L"),
      formatSecond = d3.time.format(":%S"),
      formatMinute = d3.time.format("%I:%M"),
      formatHour = d3.time.format("%I %p"),
      formatDay = d3.time.format("%a %d"),
      formatWeek = d3.time.format("%b %d"),
      formatMonth = d3.time.format("%b"),
      formatYear = d3.time.format("'%y");

  function multiFormat(date) {
    return (d3.time.second(date) < date ? formatMillisecond
        : d3.time.minute(date) < date ? formatSecond
        : d3.time.hour(date) < date ? formatMinute
        : d3.time.day(date) < date ? formatHour
        : d3.time.month(date) < date ? (d3.time.week(date) < date ? formatDay : formatWeek)
        : d3.time.year(date) < date ? formatMonth
        : formatYear)(date);
  }

  var timeLine = d3.svg.axis().scale(timeFisheye).orient("bottom").tickFormat(multiFormat)

  // Brush
  var brush = d3.svg.brush().x(xFisheye);

  brush.on("brushstart", function(){
    var xPos = d3.mouse(this)[0]
    brushStart = xPos
    dateBegin = new Date(linearTimeScale(brushStart))
    dateEnd = new Date(linearTimeScale(brushStart))
  })

  brush.on("brushend", function(){
    dateEnd = linearTimeScale(d3.mouse(this)[0])
    if (dateEnd < dateBegin) {
      var tempDate = dateEnd
      dateEnd = new Date(dateBegin)
      dateBegin = new Date(tempDate)
    }
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

  svg.on("touchmove", function() {
    var mouse = d3.mouse(this);
    xFisheye.focus(mouse[0]);
    timeFisheye.focus(mouse[0]);
    redraw();
  });

  svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0, " + timeSpaceHeight + ")")
    .call(timeLine);

  // TODO: Fisplay Date Range Values
  // TODO: Display Temporal Focus

  var tools = d3.select("body").append("div")
    .attr("id", "tools")
    .style("width", width)

  var beginText = tools.append("p"),
      endText = tools.append("p");

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
        .attr("width", Math.abs(width))

    brush.extent([x, x + width])

    beginText.text(timeFormat(dateBegin))
    
  })

  redraw();

  function redraw() {

    if (dateBegin && dateEnd) {
      var newbStart = timeFisheye(dateBegin),
          newbEnd = timeFisheye(dateEnd);

      gBrush.selectAll(".extent")
        .attr("x", newbStart)
        .attr("width", Math.abs(newbEnd - newbStart))
    }

    svg.select(".x.axis").call(timeLine);
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);
  }

})();