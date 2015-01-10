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
    .attr("class", "wake")[0][0];

  var sleep = d3.select(".day-bookend").append("input")
    .attr("type", "time")
    .attr("class", "sleep")[0][0];

  // Set Defaults

  wake.value = "09:00:00";
  sleep.value = "01:00:00";

  // Get Geolocation

  navigator.geolocation.getCurrentPosition(drawSun);

  // Prepare Canvas

  var canvas = d3.select("body").append("svg")
  .attr("width", 500)
  .attr("height", 500);

  var group = canvas.append("g")
    .attr("transform", "translate(100, 100)");

  var r = 85;
  var p = Math.PI * 2;

  var rise = moment().startOf('day');
  var fall = moment().endOf('day');

  // Construct the Proper Scale

  var time2Radians = d3.time.scale().domain([rise._d, fall._d]).range([0, p]);

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

  var dayArc = d3.svg.arc()
    .innerRadius(r - 5)
    .outerRadius(r)
    .startAngle(wakeAngleCentered)
    .endAngle(sleepAngleCentered);

  var now = d3.svg.arc()
    .innerRadius(r - 5)
    .outerRadius(r)
    .startAngle(time2Radians(moment()._d))
    .endAngle(time2Radians(moment().add(15, 'minutes')._d));

  var midnight = d3.svg.arc()
    .innerRadius(0)
    .outerRadius(r)
    .startAngle(time2Radians(moment().startOf('day')._d))
    .endAngle(time2Radians(moment().startOf('day').add(5, 'minutes')._d));

  group.append("path")
    .attr("d", dayArc)

  //group.append("path")
  //  .attr("d", midnight)

  group.append("path")
    .attr("d", now)
    .attr("class", "now")

  // Draw Sun Arc

  function threeSixty(number) {

    if (number > 360){
      number -= 360;
    }
    else if (number < 0){
      number += 360;
    }
    return number;
  }

  function drawSun(geo) {

    var lat = geo["coords"]["latitude"],
        lon = geo["coords"]["latitude"];

    var times = SunCalc.getTimes(new Date(), lat, lon),
        sunrise = moment(times['sunrise'])._d,
        sunset = moment(times['sunset']).subtract(1, 'day')._d;

    var sunArc = d3.svg.arc()
      .innerRadius(r - 10)
      .outerRadius(r + 5)
      .startAngle(time2Radians(sunrise))
      .endAngle(time2Radians(sunset));

    group.append("path")
      .attr("d", sunArc)
      .attr("class", "sun-arc")

    console.log(time2Radians(sunset))

  }

})();