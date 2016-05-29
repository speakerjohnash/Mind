(function knowledgeTree() {

	var margin = {top: 20, right: 120, bottom: 20, left: 120},
		width = 960 - margin.right - margin.left,
		height = 800 - margin.top - margin.bottom;

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

	function growTree(seedWord, json) {

		var tree = {},
			used_words = [seedWord],
			max_depth = 5,
			num_children = 5;

		if (!(seedWord in json)) {
			return 
		}

		function growBranch(word, level) {

			var children = [],
				num_twigs = num_children;

			if (level == max_depth) return;

			if (json[word].length < num_children) {
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
					branch["children"] = growBranch(cur_word, new_level)
				}

				children.push(branch)

			}

			return children

		}

		tree["name"] = seedWord
		tree["children"] = growBranch(seedWord, 0)
		tree.x0 = height / 2;
		tree.y0 = 0;

		return tree

	}

	function expandBranch(tree) {

		if (tree.children) {
			tree._children = tree.children;
			tree.children = null;
		} else {
			tree.children = tree._children;
			tree._children = null;
		}

  		updateTree(tree);

	}

	function updateTree(tree) {

		// Compute the new tree layout
		var nodes = treeLayout.nodes(treeRoot).reverse(),
			links = treeLayout.links(nodes);

		// Normalize for fixed-depth
		nodes.forEach(function(d) {d.y = d.depth * 180});

		// Update the nodes
		var node = svg.selectAll("g.node").data(nodes, function(d) {return d.id || (d.id = ++i)});

		// Enter any new nodes at the parent's previous position
		var nodeEnter = node.enter().append("g")
			.attr("class", "node")
			.attr("transform", function(d) { return "translate(" + tree.y0 + "," + tree.x0 + ")"; })
			.on("click", expandBranch);

		nodeEnter.append("circle")
			.attr("r", 1e-6)
			.style("fill", function(d) {return d._children ? "lightsteelblue" : "#fff"});

		nodeEnter.append("text")
			.attr("x", function(d) { return d.children || d._children ? -10 : 10; })
			.attr("dy", ".35em")
			.attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
			.text(function(d) { return d.name; })
			.style("fill-opacity", 1e-6);

		// Transition nodes to their new position
		var nodeUpdate = node.transition()
			.duration(duration)
			.attr("transform", function(d) {return "translate(" + d.y + "," + d.x + ")"});

		nodeUpdate.select("circle")
			.attr("r", 4.5)
			.style("fill", function(d) {return d._children ? "lightsteelblue" : "#fff"});

		nodeUpdate.select("text")
			.style("fill-opacity", 1);

		// Transition exiting nodes to the parent's new position
		var nodeExit = node.exit().transition()
			.duration(duration)
			.attr("transform", function(d) { return "translate(" + tree.y + "," + tree.x + ")"; })
			.remove();

		nodeExit.select("circle").attr("r", 1e-6);
		nodeExit.select("text").style("fill-opacity", 1e-6);

		// Update the links
		var link = svg.selectAll("path.link").data(links, function(d) {return d.target.id});

		// Enter any new links at the parent's previous position
		link.enter().insert("path", "g")
			.attr("class", "link")
			.attr("d", function(d) {
				var o = {x: tree.x0, y: tree.y0};
				return diagonal({source: o, target: o});
			});

		// Transition links to their new position
		link.transition()
			.duration(duration)
			.attr("d", diagonal);

		// Transition exiting nodes to the parent's new position
		link.exit().transition()
			.duration(duration)
			.attr("d", function(d) {
				var o = {x: tree.x, y: tree.y};
				return diagonal({source: o, target: o});
			}).remove();

		// Stash the old positions for transition
		nodes.forEach(function(d) {
			d.x0 = d.x;
			d.y0 = d.y;
		});

	}

	function visualize(word, json) {

		treeRoot = growTree(word, json)

		function collapse(tree) {
			if (tree.children) {
				tree._children = tree.children;
				tree._children.forEach(collapse);
				tree.children = null;
			}
		}

		treeRoot.children.forEach(collapse);
		updateTree(treeRoot);

	}

	function run(json) {

		d3.select("#build-tree").on("submit", function() {
			d3.event.preventDefault()
			svg.selectAll("*").remove()
			visualize(document.getElementById("seed-word").value, json)
			return false
		})

	}

	var jsonPath = "../../data/output/word2vec_tree.json";

	d3.json(jsonPath, run);

})();