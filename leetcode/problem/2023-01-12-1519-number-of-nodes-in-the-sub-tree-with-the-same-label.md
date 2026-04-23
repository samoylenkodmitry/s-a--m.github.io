---
layout: leetcode-entry
title: "1519. Number of Nodes in the Sub-Tree With the Same Label"
permalink: "/leetcode/problem/2023-01-12-1519-number-of-nodes-in-the-sub-tree-with-the-same-label/"
leetcode_ui: true
entry_slug: "2023-01-12-1519-number-of-nodes-in-the-sub-tree-with-the-same-label"
---

[1519. Number of Nodes in the Sub-Tree With the Same Label](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/description/) medium

[https://t.me/leetcode_daily_unstoppable/83](https://t.me/leetcode_daily_unstoppable/83)

[blog post](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/solutions/3039078/kotlin-build-graph-count-by-dfs/)

```kotlin
fun countSubTrees(n: Int, edges: Array<IntArray>, labels: String): IntArray {
	val graph = mutableMapOf<Int, MutableList<Int>>()
	edges.forEach { (from, to) ->
		graph.getOrPut(from, { mutableListOf() }) += to
		graph.getOrPut(to, { mutableListOf() }) += from
	}
	val answer = IntArray(n) { 0 }
	fun dfs(node: Int, parent: Int, counts: IntArray) {
		val index = labels[node].toInt() - 'a'.toInt()
		val countParents = counts[index]
		counts[index]++
		graph[node]?.forEach {
			if (it != parent) {
				dfs(it, node, counts)
			}
		}
		answer[node] = counts[index] - countParents
	}
	dfs(0, 0, IntArray(27) { 0 })
	return answer
}

```

First, we need to build a graph. Next, just do DFS and count all `'a'..'z'` frequencies in the current subtree.

For building a graph let's use a map, and for DFS let's use a recursion.
* use `parent` node instead of the visited set
* use in-place counting and subtract `count before`

Space: O(N), Time: O(N)

