---
layout: leetcode-entry
title: "2101. Detonate the Maximum Bombs"
permalink: "/leetcode/problem/2023-06-02-2101-detonate-the-maximum-bombs/"
leetcode_ui: true
entry_slug: "2023-06-02-2101-detonate-the-maximum-bombs"
---

[2101. Detonate the Maximum Bombs](https://leetcode.com/problems/detonate-the-maximum-bombs/description/) medium
[blog post](https://leetcode.com/problems/detonate-the-maximum-bombs/solutions/3587925/kotlin-directed-graph/)
[substack](https://dmitriisamoilenko.substack.com/p/02062023-2101-detonate-the-maximum?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/233
#### Problem TLDR
Count detonated bombs by chain within each radius.
#### Intuition
A bomb will only detonate if its center within the radius of another.
![image.png](/assets/leetcode_daily_images/3be78169.webp)
For example, `A` can detonate `B`, but not otherwise.

Let's build a graph, who's who can detonate.
#### Approach
Build a graph, the do DFS trying to start from each node.
#### Complexity
- Time complexity:
$$O(n^3)$$, each of the `n` DFS will take $$n^2$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun maximumDetonation(bombs: Array<IntArray>): Int {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        for (i in 0..bombs.lastIndex) {
            val bomb1 = bombs[i]
            val rr = bomb1[2] * bomb1[2].toLong()
            val edges = fromTo.getOrPut(i) { mutableListOf() }
            for (j in 0..bombs.lastIndex) {
                if (i == j) continue
                val bomb2 = bombs[j]
                val dx = (bomb1[0] - bomb2[0]).toLong()
                val dy = (bomb1[1] - bomb2[1]).toLong()
                if (dx * dx + dy * dy <= rr) edges += j
            }
        }
        fun dfs(curr: Int, visited: HashSet<Int> = HashSet()): Int {
            return if (visited.add(curr)) {
                1 + (fromTo[curr]?.sumBy { dfs(it, visited) } ?:0)
            } else 0
        }
        var max = 1
        for (i in 0..bombs.lastIndex) max = maxOf(max, dfs(i))
        return max
    }

```

