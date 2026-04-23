---
layout: leetcode-entry
title: "2642. Design Graph With Shortest Path Calculator"
permalink: "/leetcode/problem/2023-11-11-2642-design-graph-with-shortest-path-calculator/"
leetcode_ui: true
entry_slug: "2023-11-11-2642-design-graph-with-shortest-path-calculator"
---

[2642. Design Graph With Shortest Path Calculator](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/description/) hard
[blog post](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/solutions/4274939/kotlin-dijkstra/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11112023-2642-design-graph-with-shortest?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/265000b4.webp)

#### Problem TLDR

Implement graph with shortest path searching

#### Intuition

There is no special knowledge here, just a simple Dijkstra, that is BFS in a space of the shortest-so-far paths

#### Approach

* the `visited` set will improve the speed

#### Complexity

- Time complexity:
$$O(Vlog(E))$$

- Space complexity:
$$O(E)$$

#### Code

```kotlin

class Graph(n: Int, edges: Array<IntArray>) :
  HashMap<Int, MutableList<IntArray>>() {
  init { for (e in edges) addEdge(e) }

  fun addEdge(edge: IntArray) {
    getOrPut(edge[0]) { mutableListOf() } += edge
  }

  fun shortestPath(node1: Int, node2: Int): Int =
    with(PriorityQueue<Pair<Int, Int>>(compareBy({ it.second }))) {
      add(node1 to 0)
      val visited = HashSet<Int>()
      while (isNotEmpty()) {
        val (n, wp) = poll()
        if (n == node2) return@with wp
        if (visited.add(n))
          get(n)?.onEach { (_, s, w) -> add(s to (w + wp))}
      }
      -1
    }
}

```

