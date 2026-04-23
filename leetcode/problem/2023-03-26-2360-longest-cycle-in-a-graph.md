---
layout: leetcode-entry
title: "2360. Longest Cycle in a Graph"
permalink: "/leetcode/problem/2023-03-26-2360-longest-cycle-in-a-graph/"
leetcode_ui: true
entry_slug: "2023-03-26-2360-longest-cycle-in-a-graph"
---

[2360. Longest Cycle in a Graph](https://leetcode.com/problems/longest-cycle-in-a-graph/description/) hard

[blog post](https://leetcode.com/problems/longest-cycle-in-a-graph/solutions/3342651/kotlin-dfs/)

```kotlin

    fun longestCycle(edges: IntArray): Int {
        var maxLen = -1
        fun checkCycle(node: Int) {
            var x = node
            var len = 0
            do {
                if (x != edges[x]) len++
                x = edges[x]
            } while (x != node)
            if (len > maxLen) maxLen = len
        }

        val visited = HashSet<Int>()
        fun dfs(curr: Int, currPath: HashSet<Int>) {
            val isCurrentLoop = !currPath.add(curr)
            if (curr != -1 && !isCurrentLoop && visited.add(curr)) {
                dfs(edges[curr], currPath)
            } else if (curr != -1 && isCurrentLoop) checkCycle(curr)
        }
        for (i in 0..edges.lastIndex) dfs(i, HashSet<Int>())

        return maxLen
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/160
#### Intuition
We can walk all paths once and track the cycles with the DFS.

#### Approach
* Use separate visited sets for the current path and for the global visited nodes.
* Careful with `checkCycle` corner cases.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

