(function syncToSunrise() {

  // Create Input Controls

  var wake = d3.select(".day-bookend").append("input")
    .attr("type", "time")
    .attr("class", "wake")[0][0];

  var sleep = d3.select(".day-bookend").append("input")
    .attr("type", "time")
    .attr("class", "sleep")[0][0];

  // Set Defaults

  wake.value = "09:00:00";
  sleep.value = "23:00:00";

  // Draw Arc Clock

  var canvas = d3.select("body").append("svg")
    .attr("width", 500)
    .attr("height", 500);

  var group = canvas.append("g")
    .attr("transform", "translate(100, 100)");

  var r = 100;
  var p = Math.PI * 2;

  var rise = moment().startOf('day');
  var fall = moment().endOf('day');

  var time2Radians = d3.time.scale().domain([rise._d, fall._d]).range([0, p]);

  var arc = d3.svg.arc()
    .innerRadius(r - 20)
    .outerRadius(r)
    .startAngle(0)
    .endAngle(time2Radians(moment()._d));

  var wakeParts = wake.value.split(":"),
      sleepParts = sleep.value.split(":"),
      wakeAngle = time2Radians(moment().startOf('day').hour(wakeParts[0]).minute(wakeParts[1])._d),
      sleepAngle = time2Radians(moment().startOf('day').hour(sleepParts[0]).minute(sleepParts[1])._d),
      dayLength = sleepAngle - wakeAngle,
      wakeAngleCentered = -(dayLength / 2),
      sleepAngleCentered = wakeAngleCentered + dayLength;

  console.log(wakeAngle)
  console.log(sleepAngle)
  console.log(wakeAngleCentered)
  console.log(sleepAngleCentered)

  var arc2 = d3.svg.arc()
    .innerRadius((r / 2) - 5)
    .outerRadius(r / 2)
    .startAngle(wakeAngleCentered)
    .endAngle(sleepAngleCentered);

  group.append("path")
    .attr("d", arc)

  group.append("path")
    .attr("d", arc2)

})();