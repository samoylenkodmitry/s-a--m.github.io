---
layout: leetcode-entry
title: "131. Palindrome Partitioning"
permalink: "/leetcode/problem/2023-01-22-131-palindrome-partitioning/"
leetcode_ui: true
entry_slug: "2023-01-22-131-palindrome-partitioning"
---

[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/description/) medium

[https://t.me/leetcode_daily_unstoppable/93](https://t.me/leetcode_daily_unstoppable/93)

[blog post](https://leetcode.com/problems/palindrome-partitioning/solutions/3085293/kotlin-dp-and-dfs/)

```kotlin
    fun partition(s: String): List<List<String>> {
        val dp = Array(s.length) { BooleanArray(s.length) { false } }
        for (from in s.lastIndex downTo 0)
            for (to in from..s.lastIndex)
                dp[from][to] = s[from] == s[to] && (from == to || from == to - 1 || dp[from+1][to-1])
        val res = mutableListOf<List<String>>()
        fun dfs(pos: Int, partition: MutableList<String>) {
            if (pos == s.length) res += partition.toList()
            for (i in pos..s.lastIndex)
                if (dp[pos][i]) {
                    partition += s.substring(pos, i+1)
                    dfs(i+1, partition)
                    partition.removeAt(partition.lastIndex)
                }
        }
        dfs(0, mutableListOf())
        return res
    }

```

First, we need to be able to quickly tell if some range `a..b` is a palindrome.
Let's `dp[a][b]` indicate that range `a..b` is a palindrome.
Then the following is true: `dp[a][b] = s[a] == s[b] && dp[a+1][b-1]`, also two corner cases, when `a == b` and `a == b-1`.
For example, "a" and "aa".
* Use `dp` for precomputing palindrome range answers.
* Try all valid partitions with backtracking.

Space: O(2^N), Time: O(2^N)

