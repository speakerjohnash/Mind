(function knowledgeTree() {

	var margin = {top: 20, right: 120, bottom: 20, left: 120},
		width = 960 - margin.right - margin.left,
		height = 600 - margin.top - margin.bottom;

	var treeLayout = d3.layout.tree().size([height, width]),
		diagonal = d3.svg.diagonal().projection(function(d) {return [d.y, d.x]}),
		duration = 750,
		i = 0,
		treeRoot;

	var svg = d3.select(".canvas-frame").append("svg")
		.attr("width", '100%')
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	function growTree(seedWord, json, depth) {

		var weights = [],
			usedWords = [seedWord],
			maxDepth = depth,
			numChildren = 4;

		if (!(seedWord in json)) {
			return "Seed word not found. Please try another word"
		}

		function growBranch(word, level) {

			var numTwigs = numChildren,
				wordsThisLevel = [];

			if (!isNaN(word) || level <= 0 || !(word in json)) return;

			for (var i=0; i<numTwigs; i++) {

				var curWord = json[word][i]["w"],
					branch = {};

				if (usedWords.indexOf(curWord) > -1) continue;

				wordsThisLevel.push(curWord)
				usedWords.push(curWord)
				weights.push({"text": curWord, "size": level * 6 + 10})

			}

			var newLevel = level - 2

			for (var ii=0; ii<wordsThisLevel.length; ii++) {
				growBranch(wordsThisLevel[ii], newLevel)
			}

		}

		weights.push({"text": seedWord, "size": maxDepth + 10})
		growBranch(seedWord, 16)

		shuffle(weights)
		return weights

	}

	function updateTree(wordList) {

		console.log(wordList)

		// TODO Weight to Font Size Scale

		// TODO Colors

		var cloudLayout = d3.layout.cloud().size([800, 600])
            .words(wordList)
            .rotate(0)
            .fontSize(function(d) { return d.size; })
            .on("end", draw)
            .start();

        function draw(words) {
        	svg.append("g")
        		.attr("transform", "translate(" + 200 + "," + 200 + ")")
        		.selectAll("text")
     			.data(words)
				.enter().append("text")
      			.style("font-size", function(d) { return d.size + "px"; })
      			.attr("text-anchor", "middle")
      			.attr("transform", function(d) {
                    return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                })
      			.text(function(d) { return d.text; });
        	console.log(words)
        }

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

	function visualize(word, json) {

		wordList = growTree(word, json)

		updateTree(wordList);

	}

	function run(json) {

		d3.select("#build-tree").on("submit", function() {
			d3.event.preventDefault()
			svg.selectAll("*").remove()
			visualize(document.getElementById("seed-word").value, json)
			return false
		})

		treeRoot = growTree("word2vec", json, 3)
		updateTree(treeRoot);

	}

	var jsonPath = "../../data/output/fruiting_word2vec_tree.json";

	d3.json(jsonPath, run);

})();