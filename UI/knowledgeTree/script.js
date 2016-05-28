(function knowledgeTree() {

	function growTree(seedWord, json) {

		var tree = {},
			used_words = [seedWord],
			max_depth = 5,
			num_branches = 5;

		if (!(seedWord in json)) {
			return 
		}

		function growBranch(word, level) {

			var branches = [],
				num_twigs = num_branches;

			if (level == max_depth) return;

			if (json[word].length < num_branches) {
				num_twigs = json[word].length
			}

			for (var i=0; i<num_twigs; i++) {

				var cur_word = json[word][i]["w"],
					branch = {},
					new_level = level + 1;

				if (used_words.indexOf(cur_word) > -1) continue;

				used_words.push(cur_word)
				branch["name"] = cur_word

				if (new_level <= max_depth) {
					branch["branches"] = buildFlare(cur_word, new_level)
				}

				branches.push(brnach)

			}

			return branches

		}

		tree["name"] = seedWord
		tree["branches"] = buildFlare(seedWord, 0)

		return tree

	}

	function visualize(word, tree, json) {

		console.log(json)

	}

	function createSVG(json) {

		var margin = {top: 20, right: 120, bottom: 20, left: 120},
			width = 960 - margin.right - margin.left,
			height = 800 - margin.top - margin.bottom;

		var tree = d3.layout.tree().size([height, width]),
			diagonal = d3.svg.diagonal().projection(function(d) { return [d.y, d.x]; });

		var svg = d3.select(".canvas-frame").append("svg")
			.attr("width", '100%')
			.attr("height", height + margin.top + margin.bottom)
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

		d3.select("#build-tree").on("submit", function() {
			d3.event.preventDefault()
			svg.selectAll("*").remove()
			visualize(document.getElementById("seed-word").value, tree, json)
			return false
		})

	}

	var jsonPath = "../../data/output/word2vec_tree.json";

	d3.json(jsonPath, createSVG);

})();