---
layout: leetcode-entry
title: "1857. Largest Color Value in a Directed Graph"
permalink: "/leetcode/problem/2023-04-09-1857-largest-color-value-in-a-directed-graph/"
leetcode_ui: true
entry_slug: "2023-04-09-1857-largest-color-value-in-a-directed-graph"
---

[1857. Largest Color Value in a Directed Graph](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/description/) hard

[blog post](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/solutions/3396443/kotlin-dfs-cache/)

```kotlin

fun largestPathValue(colors: String, edges: Array<IntArray>): Int {
    if (edges.isEmpty()) return if (colors.isNotEmpty()) 1 else 0
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> fromTo.getOrPut(from) { mutableListOf() } += to }
        val cache = mutableMapOf<Int, IntArray>()
        var haveCycle = false
        fun dfs(curr: Int, visited: HashSet<Int> = HashSet()): IntArray {
            return cache.getOrPut(curr) {
                val freq = IntArray(26)
                if (visited.add(curr)) {
                    fromTo.remove(curr)?.forEach {
                        val childFreq = dfs(it, visited)
                        for (i in 0..25) freq[i] = maxOf(childFreq[i], freq[i])
                    }
                    freq[colors[curr].toInt() - 'a'.toInt()] += 1
                } else haveCycle = true
                freq
            }
        }
        var max = 0
        edges.forEach { (from, to) -> max = maxOf(max, dfs(from).max()!!) }
        return if (haveCycle) -1 else max
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/175
#### Intuition
![image.png](/assets/leetcode_daily_images/0666f73f.webp)
![leetcode_daily_small.gif](/assets/leetcode_daily_images/5c8c31ed.webp)

For each node, there is only one answer of the maximum count of the same color. For each parent, $$c_p = max(freq_{child})+colors[curr]$$. We can cache the result and compute it using DFS and selecting maximum count from all the children.
#### Approach
* use `visited` set to detect cycles
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

