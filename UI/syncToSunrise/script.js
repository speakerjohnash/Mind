// Converts from degrees to radians

Math.toRadians = function(degrees) {
  return degrees * Math.PI / 180;
};

// Converts from radians to degrees

Math.toDegrees = function(radians) {
  return radians * 180 / Math.PI;
};
 
(function syncToSunrise() {

  // Create Input Controls

  var wake = d3.select(".day-bookend").append("input")
    .attr("type", "time")
    .attr("class", "wake")
    .on("change", drawClock)[0][0];

  var sleep = d3.select(".day-bookend").append("input")
    .attr("type", "time")
    .attr("class", "sleep")
    .on("change", drawClock)[0][0];

  // Set Defaults

  wake.value = "08:30:00";
  sleep.value = "01:30:00";

  // Prepare Canvas

  var px_diameter = 650;

  var canvas = d3.select(".clock-wrapper")
    .style("width", px_diameter + "px")
    .append("svg")
    .attr("width", px_diameter)
    .attr("height", px_diameter);

  var filter = canvas.append("defs")
    .append("filter")
    .attr("id", "blur")
    .append("feGaussianBlur")
    .attr("stdDeviation", 2);

  var group = canvas.append("g")
    .attr("transform", "translate(" + (px_diameter / 2) + "," + (px_diameter / 2) + ")");

  var r = 3/7 * px_diameter;
  var p = Math.PI * 2;

  var rise = moment().startOf('day');
  var fall = moment().endOf('day');

  // Draw a Clock
  drawClock()
  setInterval(drawClock, 1000);

  function drawClock() {

    // Construct the Proper Scale

    var time2Radians = d3.time.scale().domain([rise._d, fall._d]).range([0, p]);

    group.selectAll("*").remove();

    var wakeParts = wake.value.split(":"),
      sleepParts = sleep.value.split(":"),
      wakeTime = moment().startOf('day').hour(wakeParts[0]).minute(wakeParts[1]),
      sleepTime = moment().startOf('day').hour(sleepParts[0]).minute(sleepParts[1]),
      sleepTime = (sleepTime.isBefore(wakeTime)) ? sleepTime.add(1, 'day') : sleepTime,
      wakeAngle = time2Radians(wakeTime._d),
      sleepAngle = time2Radians(sleepTime._d),
      dayLength = sleepAngle - wakeAngle,
      wakeAngleCentered = -(dayLength / 2),
      sleepAngleCentered = wakeAngleCentered + dayLength,
      zeroAngle = wakeAngleCentered - wakeAngle;

    // The Magic Scale that Converts between Linear Time and Arc Time

    time2Radians.range([zeroAngle, zeroAngle + p])

    // Construct and Draw Arcs

    var lineWidth = r * 0.0833;

    var dayArc = d3.svg.arc()
      .innerRadius(r - lineWidth)
      .outerRadius(r)
      .startAngle(wakeAngleCentered)
      .endAngle(sleepAngleCentered);

    var sleepArc = d3.svg.arc()
      .innerRadius(r - lineWidth)
      .outerRadius(r)
      .startAngle(sleepAngleCentered)
      .endAngle(wakeAngleCentered + p);

    var now = d3.svg.arc()
      .innerRadius(r - lineWidth)
      .outerRadius(r)
      .startAngle(time2Radians(moment()._d))
      .endAngle(time2Radians(moment().add(15, 'minutes')._d));

    var backing = group.append("g")
    
    backing.append("circle")
      .attr("class", "back")
      .attr("r", r + (2 * lineWidth))
    
    backing.append("text")
      .attr("text-anchor", "middle")
      .attr("class", "quicksand digital-time")
      .style("font-size", (r / 3) + "px")
      .text(moment().format("h:mm a")); 
    
    group.append("path")
      .attr("d", dayArc)
      .attr("class", "day-arc")

    group.append("path")
      .attr("d", sleepArc)
      .attr("class", "sleep-arc")

    group.append("path")
      .attr("d", now)
      .attr("class", "now")

    // Get Geolocation and Draw Sun Arc

    var lat = Cookies.get("latitude"),
        lon = Cookies.get("longitude");

    if (typeof(lat) == "undefined" || typeof(lon) == "undefined") {
      navigator.geolocation.getCurrentPosition(drawSun);
    } else {
      drawSun({"coords":{"latitude": lat, "longitude": lon}})
    }

    function drawSun(geo) {

      var lat = geo["coords"]["latitude"],
          lon = geo["coords"]["longitude"];

      var times = SunCalc.getTimes(new Date(), lat, lon),
          sunrise = moment(times['sunrise'])._d,
          sunset = moment(times['sunset'])._d;

      var sunArc = d3.svg.arc()
        .innerRadius(r - (1.5 * lineWidth))
        .outerRadius(r + (.5 * lineWidth))
        .startAngle(time2Radians(sunrise))
        .endAngle(time2Radians(sunset));

      group.append("path")
        .attr("d", sunArc)
        .attr("class", "sun-arc")
        .attr("filter", "url(#blur)");

      Cookies.set("latitude", lat);
      Cookies.set("longitude", lon)

    }

  }

})();