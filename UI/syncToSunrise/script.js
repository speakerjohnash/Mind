(function syncToSunrise() {
  
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

  group.append("path")
    .attr("d", arc)

})();