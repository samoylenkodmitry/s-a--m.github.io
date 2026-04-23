---
layout: leetcode-entry
title: "785. Is Graph Bipartite?"
permalink: "/leetcode/problem/2023-05-19-785-is-graph-bipartite/"
leetcode_ui: true
entry_slug: "2023-05-19-785-is-graph-bipartite"
---

[785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/description/) medium
[blog post](https://leetcode.com/problems/is-graph-bipartite/solutions/3540319/kotlin-dfs-red-blue/)
[substack](https://dmitriisamoilenko.substack.com/p/19052023-785-is-graph-bipartite?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/217
#### Problem TLDR
Find if graph is [bipartite](https://en.wikipedia.org/wiki/Bipartite_graph)
#### Intuition
![image.png](/assets/leetcode_daily_images/21d97c3d.webp)
Mark edge `Red` or `Blue` and it's nodes in the opposite.

#### Approach
* there are disconnected nodes, so run DFS for all of them
#### Complexity
- Time complexity:
$$O(VE)$$, DFS once for all `vertices` and `edges`
- Space complexity:
$$O(V+E)$$, for `reds` and `visited` set.

#### Code

```kotlin

fun isBipartite(graph: Array<IntArray>): Boolean {
    val reds = IntArray(graph.size)
    fun dfs(u: Int, isRed: Int): Boolean {
        if (reds[u] == 0) {
            reds[u] = if (isRed == 0) 1 else isRed
            return graph[u].all { dfs(it, -reds[u]) }
        } else return reds[u] == isRed
    }
    return graph.indices.all { dfs(it, reds[it]) }
}

```

