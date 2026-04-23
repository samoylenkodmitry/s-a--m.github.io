---
layout: leetcode-entry
title: "1514. Path with Maximum Probability"
permalink: "/leetcode/problem/2023-06-28-1514-path-with-maximum-probability/"
leetcode_ui: true
entry_slug: "2023-06-28-1514-path-with-maximum-probability"
---

[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/description/) medium
[blog post](https://leetcode.com/problems/path-with-maximum-probability/solutions/3691288/kotlin-dijkstra/)
[substack](https://dmitriisamoilenko.substack.com/p/28062023-1514-path-with-maximum-probability?sd=pf)
![image.png](/assets/leetcode_daily_images/f27379f4.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/259
#### Problem TLDR
Max probability path from `start` to `end` in a probability edges graph
#### Intuition
What didn't work:
* naive BFS, DFS with `visited` set - will not work, as we need to visit some nodes several times
* Floyd-Warshall - will solve this problem for every pair of nodes, but takes $$O(n^3)$$ and gives TLE
What will work: Dijkstra
#### Approach
* store probabilities from `start` to every node in an array
* the stop condition will be when there is no any `better` path

#### Complexity

- Time complexity:
$$O(EV)$$

- Space complexity:
$$O(EV)$$

#### Code

```kotlin

fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start: Int, end: Int): Double {
    val pstart = Array(n) { 0.0 }
    val adj = mutableMapOf<Int, MutableList<Pair<Int, Double>>>()
    edges.forEachIndexed { i, (from, to) ->
        adj.getOrPut(from) { mutableListOf() } += to to succProb[i]
        adj.getOrPut(to) { mutableListOf() } += from to succProb[i]
    }
    with(ArrayDeque<Pair<Int, Double>>()) {
        add(start to 1.0)
        while(isNotEmpty()) {
            val (curr, p) = poll()
            if (p <= pstart[curr]) continue
            pstart[curr] = p
            adj[curr]?.forEach { (next, pnext) -> add(next to p * pnext) }
        }
    }

    return pstart[end]
}

```

