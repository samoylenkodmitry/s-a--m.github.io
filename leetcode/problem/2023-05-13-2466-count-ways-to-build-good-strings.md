---
layout: leetcode-entry
title: "2466. Count Ways To Build Good Strings"
permalink: "/leetcode/problem/2023-05-13-2466-count-ways-to-build-good-strings/"
leetcode_ui: true
entry_slug: "2023-05-13-2466-count-ways-to-build-good-strings"
---

[2466. Count Ways To Build Good Strings](https://leetcode.com/problems/count-ways-to-build-good-strings/description/) medium
[blog post](https://leetcode.com/problems/count-ways-to-build-good-strings/solutions/3518102/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/13052023-2466-count-ways-to-build?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/211

#### Problem
Count distinct strings, length low to high, appending '0' zero or '1' one times. Return count % 1,000,000,007.

#### Intuition
Let's add `zero`'s or `one`'s one by one. For each current length, the resulting count is independent of all the previous additions. We can cache the result by the current `size` of the string.

#### Approach
Let's write a DFS solution, adding `zero` or `one` and count the good strings.
Then we can rewrite it to the iterative DP.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code
top-down:

```

fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
    val m = 1_000_000_007
    val cache = mutableMapOf<Int, Int>()
    fun dfs(currSize: Int): Int {
        if (currSize > high) return 0
        return cache.getOrPut(currSize) {
            val curr = if (currSize in low..high) 1 else 0
            val addZeros = if (zero > 0) dfs(currSize + zero) else 0
            val addOnes = if (one > 0) dfs(currSize + one) else 0
            (curr + addZeros + addOnes) % m
        }
    }
    return dfs(0)
}

```

bottom-up

```

fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
    val cache = mutableMapOf<Int, Int>()
    for (sz in high downTo 0)
    cache[sz] = ((if (sz >= low) 1 else 0)
    + (cache[sz + zero]?:0)
    + (cache[sz + one]?:0)) % 1_000_000_007
    return cache[0]!!
}

```

