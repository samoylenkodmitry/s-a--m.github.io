---
layout: leetcode-entry
title: "87. Scramble String"
permalink: "/leetcode/problem/2023-03-30-87-scramble-string/"
leetcode_ui: true
entry_slug: "2023-03-30-87-scramble-string"
---

[87. Scramble String](https://leetcode.com/problems/scramble-string/description/) hard

[blog post](https://leetcode.com/problems/scramble-string/solutions/3358175/kotlin-dfs-memo-no-substring/)

```kotlin

data class Key(val afrom: Int, val ato: Int, val bfrom: Int, val bto: Int)
fun isScramble(a: String, b: String): Boolean {
    val dp = HashMap<Key, Boolean>()
    fun dfs(key: Key): Boolean {
        return dp.getOrPut(key) {
            val (afrom, ato, bfrom, bto) = key
            val alength = ato - afrom
            val blength = bto - bfrom
            if (alength != blength) return@getOrPut false
            var same = true
            for (i in 0..alength)
            if (a[afrom + i] != b[bfrom + i]) same = false
            if (same) return@getOrPut true
            for (i in afrom..ato - 1) {
                if (dfs(Key(afrom, i, bfrom, bfrom + (i - afrom)))
                && dfs(Key(i + 1, ato, bfrom + (i - afrom) + 1, bto))) return@getOrPut true
                if (dfs(Key(afrom, i, bto - (i - afrom), bto))
                && dfs(Key(i + 1, ato, bfrom, bto - (i - afrom) - 1))) return@getOrPut true
            }

            return@getOrPut false
        }
    }
    return dfs(Key(0, a.lastIndex, 0, b.lastIndex))
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/164
#### Intuition
This is not a permutation's problem, as there are examples when we can't scramble two strings consisting of the same characters.
We can simulate the process and search the result using DFS.

#### Approach
A simple approach is to concatenate strings, but in Kotlin it gives TLE, so we need bottom up approach, or just operate with indices.
* use including indices ranges
* in Kotlin, don't forget `@getOrPut` when exiting lambda
#### Complexity
- Time complexity:
$$O(n^4)$$
- Space complexity:
$$O(n^4)$$

