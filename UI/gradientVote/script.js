gradientVote = function (labels) {

  var width = window.innerWidth,
    widgetWidth = 175,
    height = 50,
    radius = 10,
    margin = 25
    leftLabel = labels[0],
    rightLabel = labels[1];

  var x1 = margin,
    x2 = widgetWidth + margin,
    y = height / 2;

  var container = d3.select("body")
    .append("div")
    .attr("id", "gradient-container")
    .style("width", widgetWidth + (2 * margin) + "px")

  var labels = container.append("div")
    .attr("class", "labels")
    .selectAll("span")
    .data(labels)
    .enter()
    .append("span")
    .text(function(d) { return d })
    .attr("class", function(d, i) { return "label-" + i})

  var svg = container.append("svg")
    .attr("width", widgetWidth + (2 * margin))
    .attr("height", height)
    .on("mousemove", hoverMove)
    .datum({
      x: (widgetWidth / 2) + margin,
      y: height / 2
    });

  var empty = svg.append("line")
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

}(["False", "True"])