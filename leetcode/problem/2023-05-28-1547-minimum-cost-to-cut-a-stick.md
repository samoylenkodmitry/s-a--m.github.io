---
layout: leetcode-entry
title: "1547. Minimum Cost to Cut a Stick"
permalink: "/leetcode/problem/2023-05-28-1547-minimum-cost-to-cut-a-stick/"
leetcode_ui: true
entry_slug: "2023-05-28-1547-minimum-cost-to-cut-a-stick"
---

[1547. Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/submissions/958762191/) hard
[blog post](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/solutions/3570530/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/28052023-1547-minimum-cost-to-cut?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/226
#### Problem TLDR
Min cost of cuts `c1,..,ci,..,cn` of `[0..n]` where cut cost the `length = to-from`.
#### Intuition
We every stick `from..to` we can try all the cuts in that range. This result will be optimal and can be cached.

#### Approach
* use DFS + memo
* check for range
#### Complexity
- Time complexity:
$$k^2$$, as maximum depth of DFS is `k`, and we loop for `k`.
- Space complexity:
$$k^2$$

#### Code

```kotlin

fun minCost(n: Int, cuts: IntArray): Int {
    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(from: Int, to: Int): Int {
        return cache.getOrPut(from to to) {
            var min = 0
            cuts.forEach {
                if (it in from + 1..to - 1) {
                    val new = to - from + dfs(from, it) + dfs(it, to)
                    if (min == 0 || new < min) min = new
                }
            }

            min
        }
    }
    return dfs(0, n)
}

```

