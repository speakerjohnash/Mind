(function timeSelect() {
  var width = window.innerWidth / 1.45,
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
      timeFisheye = d3.fisheye.scale(d3.scaleTime).domain([now, then]).range([0, width]).focus(0),
      timeFormat = d3.timeFormat("%m/%d/%Y");

  var xFisheye = d3.fisheye.scale(d3.scaleIdentity).domain([0, width]).focus(0);

  var chartContainer = d3.select("#chart")
    .style("width", width + 40 + "px");

  var linearTimeScale = d3.scaleTime().domain([0, width]).range([now, then]);

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
  var formatMillisecond = d3.timeFormat(".%L"),
      formatSecond = d3.timeFormat(":%S"),
      formatMinute = d3.timeFormat("%I:%M"),
      formatHour = d3.timeFormat("%I %p"),
      formatDay = d3.timeFormat("%a %d"),
      formatWeek = d3.timeFormat("%b %d"),
      formatMonth = d3.timeFormat("%b"),
      formatYear = d3.timeFormat("'%y");

  function multiFormat(date) {
    return (d3.timeSecond(date) < date ? formatMillisecond
        : d3.timeMinute(date) < date ? formatSecond
        : d3.timeHour(date) < date ? formatMinute
        : d3.timeDay(date) < date ? formatHour
        : d3.timeMonth(date) < date ? (d3.timeWeek(date) < date ? formatDay : formatWeek)
        : d3.timeYear(date) < date ? formatMonth
        : formatYear)(date);
  }

  var timeLine = d3.axisBottom().scale(timeFisheye).tickFormat(multiFormat)

  // Brush
  var brush = d3.brushX(xFisheye);

  brush.on("start", function(){
    var xPos = d3.mouse(this)[0]
    brushStart = xPos
    dateBegin = new Date(linearTimeScale(brushStart))
    dateEnd = new Date(linearTimeScale(brushStart))
  })

  brush.on("end", function(){
    dateEnd = new Date(linearTimeScale(d3.mouse(this)[0]))
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

  brush.on("brush", brushed)

  redraw();

  function redraw() {

    if (dateBegin && dateEnd) {
      var newbStart = timeFisheye(dateBegin),
          newbEnd = timeFisheye(dateEnd);

      gBrush.selectAll(".selection")
        .attr("x", newbStart)
        .attr("width", Math.abs(newbEnd - newbStart))

    }

    svg.select(".x.axis").call(timeLine);
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);

  }

  function brushed() {

    if (d3.event.sourceEvent.type === "brush") return;

    var xMouse = d3.mouse(this)[0],
        x,
        width;

    timeFisheye.focus(xMouse);
    xFisheye.focus(xMouse);

    var newPixel = timeFisheye(dateBegin);

    if (xMouse > brushStart) {
      x = newPixel
      width = xMouse - newPixel
    } else {
      x = xMouse
      width = newPixel - xMouse
    }

    d3.select(this).call(brush.move, [x, x + width]);
    svg.select(".x.axis").call(timeLine);
    xLine.attr("x1", xFisheye).attr("x2", xFisheye);
    beginText.text("Begin: " + timeFormat(dateBegin))
    endText.text("End: " + timeFormat(new Date(linearTimeScale(xMouse))))

  }

})();