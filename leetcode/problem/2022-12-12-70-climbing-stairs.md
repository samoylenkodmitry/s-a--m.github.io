---
layout: leetcode-entry
title: "70. Climbing Stairs"
permalink: "/leetcode/problem/2022-12-12-70-climbing-stairs/"
leetcode_ui: true
entry_slug: "2022-12-12-70-climbing-stairs"
---

[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/) easy

[https://t.me/leetcode_daily_unstoppable/49](https://t.me/leetcode_daily_unstoppable/49)

[blog post](https://leetcode.com/problems/climbing-stairs/solutions/2904774/kotlin-dfs-memo/)

```kotlin
    val cache = mutableMapOf<Int, Int>()
    fun climbStairs(n: Int): Int = when {
        n < 1  -> 0
        n == 1 -> 1
        n == 2 -> 2
        else -> cache.getOrPut(n) {
            climbStairs(n-1) + climbStairs(n-2)
        }
    }

```

You can observe that result is only depend on input n. And also that result(n) = result(n-1) + result(n-2).
Just use memoization for storing already solved inputs.

Space: O(N), Time: O(N)

