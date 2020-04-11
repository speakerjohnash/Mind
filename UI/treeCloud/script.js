(function knowledgeTree() {

	var margin = {top: 20, right: 120, bottom: 20, left: 120},
		width = 960 - margin.right - margin.left,
		height = 600 - margin.top - margin.bottom,
		fontName = 'Quicksand',
		fruitColor = "#7599c3",
		colorPallet = ["#443462", "#455a8b", "#457a8b"],
		globalJSON = {};

	var duration = 750,
		i = 0,
		treeRoot;

	var cloudLayout = d3.layout.cloud().size([width, height])
		.padding(5)
		.rotate(0)
		.font(fontName)
		.fontSize(function(d) { return d.size; })
		.on("end", draw);

	var color = d3.scale.linear().range(colorPallet);

	var svg = d3.select(".canvas-frame").append("svg")
		.attr("width", '100%')
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")")
		.attr("width", cloudLayout.size()[0])
		.attr("height", cloudLayout.size()[1]);

	function draw(words, bounds) {
		svg.append("g")
			.attr("transform", "translate(" + cloudLayout.size()[0] / 2 + "," + cloudLayout.size()[1] / 2 + ")")
			.selectAll("text")
			.data(words)
			.enter().append("text")
			.style("font-size", function(d) { return d.size + "px"; })
			.style("fill", function(d) { return d.fruit ? "black" : fruitColor })
			.attr("font-family", fontName)
			.attr("text-anchor", "middle")
			.style("cursor", function(d) {
				if (!d.fruit) {
					return "context-menu"
				} else {
					return "pointer"
				}
			})
			.on("click", function(d){
				if (!d.fruit) {
					var url = "https://www.google.com/search?q=" + d.text + "+definition"
					window.open(url, '_blank');
					return
				}
				var word = d.text
				visualize(word)
			})
			.attr("transform", function(d) {
				return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
			})
			.text(function(d) { return d.text; });
	}

	function growTree(seedWord, depth) {

		var weights = [],
			usedWords = [seedWord],
			maxDepth = depth,
			numChildren = 3;

		if (!(seedWord in globalJSON)) {
			return "Seed word not found. Please try another word"
		}

		function growBranch(word, level) {

			var numTwigs = numChildren,
				wordsThisLevel = [];

			if (!isNaN(word) || level <= 0 || !(word in globalJSON)) return;

			for (var i=0; i<numTwigs; i++) {

				if (i == globalJSON[word].length) break;

				var curWord = globalJSON[word][i]["w"],
				branch = {};

				if (usedWords.indexOf(curWord) > -1) continue;

				wordsThisLevel.push(curWord)
				usedWords.push(curWord)
				weights.push({
					"text": curWord, 
					"size": level > 6 ? level * 2 : 6,
					"fruit": globalJSON[word][i]["l"]
				})

			}

			var newLevel = level - 2

			for (var ii=0; ii<wordsThisLevel.length; ii++) {
				numChildren = Math.floor(Math.random() * 4) + 2
				growBranch(wordsThisLevel[ii], newLevel)
			}

		}

		weights.push({"text": seedWord, "size": 85, "fruit": true})
		growBranch(seedWord, 16)

		shuffle(weights)
		return weights

	}

	function shuffle(array) {

		var currentIndex = array.length, temporaryValue, randomIndex;

		while (0 !== currentIndex) {
			randomIndex = Math.floor(Math.random() * currentIndex);
			currentIndex -= 1;
			temporaryValue = array[currentIndex];
			array[currentIndex] = array[randomIndex];
			array[randomIndex] = temporaryValue;
		}

		return array;

	}

	function visualize(word) {
		svg.selectAll("*").remove()
		wordList = growTree(word, 5)
		max = d3.max(wordList, function(d) { return d.size; })
		min = d3.min(wordList, function(d) { return d.size; })
		color.domain([0, (max / 4), max])
		cloudLayout.words(wordList).start();
	}

	function run(json) {

		globalJSON = json

		d3.select("#build-tree").on("submit", function() {
			d3.event.preventDefault()
			visualize(document.getElementById("seed-word").value)
			return false
		})

		var words = Object.keys(json),
			randomWord = words[Math.floor(Math.random()*words.length)];

		visualize(randomWord)

	}

	var jsonPath = "../../data/sensemaking_word2vec_tree_03-31-20.json";

	d3.json(jsonPath, run);

})();