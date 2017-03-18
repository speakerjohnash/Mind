gradientVote = function () {

  var width = window.innerWidth,
    widgetWidth = 175,
    height = 500,
    radius = 10,
    margin = 100;

  var x1 = margin,
    x2 = widgetWidth + margin,
    y = height / 2;

  var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .on("mousemove", hoverMove)
    .datum({
      x: (widgetWidth / 2) + margin,
      y: height / 2
    });

  var container = svg.append("line")
    .attr("x1", x1)
    .attr("x2", x2)
    .attr("y1", y)
    .attr("y2", y)
    .style("stroke", "#f2f2f2")
    .style("stroke-linecap", "round")
    .style("stroke-width", radius * 2);

  var truth = svg.append("line")
    .attr("x1", x1)
    .attr("x2", function(d) { return d.x; })
    .attr("y1", y)
    .attr("y2", y)
    .style("stroke", "#f1e886")
    .style("stroke-linecap", "round")
    .style("stroke-width", radius * 2);

  function hoverMove(d) {

    var x = d3.event.x;

    // Constrain x to be between x1 and x2 (the ends of the line).
    x = x < x1 ? x1 : x > x2 ? x2 : x;

    truth.attr("x2", x);

  }

}()