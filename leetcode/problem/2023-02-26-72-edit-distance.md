---
layout: leetcode-entry
title: "72. Edit Distance"
permalink: "/leetcode/problem/2023-02-26-72-edit-distance/"
leetcode_ui: true
entry_slug: "2023-02-26-72-edit-distance"
---

[72. Edit Distance](https://leetcode.com/problems/edit-distance/description/) hard

[blog post](https://leetcode.com/problems/edit-distance/solutions/3231899/kotlin-dfs-memo/)

```kotlin

fun minDistance(word1: String, word2: String): Int {
    val dp = Array(word1.length + 1) { IntArray(word2.length + 1) { -1 } }
    fun dfs(i: Int, j: Int): Int {
        return when {
            dp[i][j] != -1 -> dp[i][j]
            i == word1.length && j == word2.length -> 0
            i == word1.length -> 1 + dfs(i, j+1)
            j == word2.length -> 1 + dfs(i+1, j)
            word1[i] == word2[j] -> dfs(i+1, j+1)
            else -> {
                val insert1Delete2 = 1 + dfs(i, j+1)
                val insert2Delete1 = 1 + dfs(i+1, j)
                val replace1Or2 = 1 + dfs(i+1, j+1)
                val res = minOf(insert1Delete2, insert2Delete1, replace1Or2)
                dp[i][j] = res
                res
            }
        }
    }
    return dfs(0, 0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/130
#### Intuition
Compare characters from each positions of the two strings. If they are equal, do nothing. If not, we can choose from three paths: removing, inserting or replacing. That will cost us `one` point of operations. Then, do DFS and choose the minimum of the operations.

#### Approach
Do DFS and use array for memoizing the result.
#### Complexity
- Time complexity:
$$O(n^2)$$, can be proven if you rewrite DP to bottom up code.
- Space complexity:
$$O(n^2)$$

