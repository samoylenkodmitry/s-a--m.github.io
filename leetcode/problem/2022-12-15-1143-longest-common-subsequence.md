---
layout: leetcode-entry
title: "1143. Longest Common Subsequence"
permalink: "/leetcode/problem/2022-12-15-1143-longest-common-subsequence/"
leetcode_ui: true
entry_slug: "2022-12-15-1143-longest-common-subsequence"
---

[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description/) medium

[https://t.me/leetcode_daily_unstoppable/52](https://t.me/leetcode_daily_unstoppable/52)

[blog post](https://leetcode.com/problems/longest-common-subsequence/solutions/2915134/kotlin-dfs-memo/)

```kotlin
    fun longestCommonSubsequence(text1: String, text2: String): Int {
        val cache = Array(text1.length + 1) { IntArray(text2.length + 1) { -1 } }
        fun dfs(pos1: Int, pos2: Int): Int {
            if (pos1 == text1.length) return 0
            if (pos2 == text2.length) return 0
            val c1 = text1[pos1]
            val c2 = text2[pos2]
            if (cache[pos1][pos2] != -1) return cache[pos1][pos2]
            val res = if (c1 == c2) {
                    1 + dfs(pos1 + 1, pos2 + 1)
                } else {
                    maxOf(dfs(pos1, pos2+1), dfs(pos1+1, pos2))
                }
            cache[pos1][pos2] = res
            return res
        }
        return dfs(0, 0)
    }

```

We can walk the two strings simultaneously and compare their chars. If they are the same, the optimal way will be to use those chars and continue exploring next. If they are not, we have two choices: use the first char and skip the second or skip the first but use the second.
Also, observing our algorithm we see, the result so far is only dependent of the positions from which we begin to search (and all the remaining characters). And also see that the calls are repetitive. That mean we can cache the result. (meaning this is a dynamic programming solution).
Use depth first search by starting positions and memoize results in a two dimension array. Another approach will be bottom up iteration and filling the same array.

Space: O(N^2), Time: O(N^2)

