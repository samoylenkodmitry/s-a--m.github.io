---
layout: leetcode-entry
title: "1626. Best Team With No Conflicts"
permalink: "/leetcode/problem/2023-01-31-1626-best-team-with-no-conflicts/"
leetcode_ui: true
entry_slug: "2023-01-31-1626-best-team-with-no-conflicts"
---

[1626. Best Team With No Conflicts](https://leetcode.com/problems/best-team-with-no-conflicts/description/) medium

[blog post](https://leetcode.com/problems/best-team-with-no-conflicts/solutions/3123505/kotlin-dfs-memo/)

```kotlin
    fun bestTeamScore(scores: IntArray, ages: IntArray): Int {
        val dp = Array(scores.size + 1) { IntArray(1001) { -1 }}
        val indices = scores.indices.toMutableList()
        indices.sortWith(compareBy( { scores[it] }, { ages[it] } ))
        fun dfs(curr: Int, prevAge: Int): Int {
            if (curr == scores.size) return 0
            if (dp[curr][prevAge] != -1) return dp[curr][prevAge]
            val ind = indices[curr]
            val age = ages[ind]
            val score = scores[ind]
            val res = maxOf(
                dfs(curr + 1, prevAge),
                if (age < prevAge) 0  else score + dfs(curr + 1, age)
            )
            dp[curr][prevAge] = res
            return res
        }
        return dfs(0, 0)
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/103
#### Intuition
If we sort arrays by `score` and `age`, then every next item will be with  `score` bigger than previous.
If current `age` is less than previous, then we can't take it, as `score` for current `age` can't be bigger than previous.
Let's define `dp[i][j]` is a maximum score for a team in `i..n` sorted slice, and `j` is a maximum age for that team.
#### Approach
We can use DFS to search all the possible teams and memorize the result in dp cache.
#### Complexity
- Time complexity:
  $$O(n^2)$$, we can only visit n by n combinations of pos and age
- Space complexity:
  $$O(n^2)$$

