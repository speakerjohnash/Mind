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

	function processData(error, json) {

	}

	var jsonPath = "../../data/output/word2vec_tree.json";
	
	d3.csv(csvpath, processData);

})();