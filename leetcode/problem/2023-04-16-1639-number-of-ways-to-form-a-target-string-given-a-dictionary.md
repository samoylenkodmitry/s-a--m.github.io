---
layout: leetcode-entry
title: "1639. Number of Ways to Form a Target String Given a Dictionary"
permalink: "/leetcode/problem/2023-04-16-1639-number-of-ways-to-form-a-target-string-given-a-dictionary/"
leetcode_ui: true
entry_slug: "2023-04-16-1639-number-of-ways-to-form-a-target-string-given-a-dictionary"
---

[1639. Number of Ways to Form a Target String Given a Dictionary](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/) hard

```kotlin

fun numWays(words: Array<String>, target: String): Int {
    val freq = Array(words[0].length) { LongArray(26) }
    for (i in 0..words[0].lastIndex)
    words.forEach { freq[i][it[i].toInt() - 'a'.toInt()]++ }

    val cache = Array(words[0].length) { LongArray(target.length) { -1L } }
    val m = 1_000_000_007L

    fun dfs(wpos: Int, tpos: Int): Long {
        if (tpos == target.length) return 1L
        if (wpos == words[0].length) return 0L
        if (cache[wpos][tpos] != -1L) return cache[wpos][tpos]
        val curr = target[tpos].toInt() - 'a'.toInt()
        val currFreq = freq[wpos][curr]
        val take = if (currFreq == 0L) 0L else
        dfs(wpos + 1, tpos + 1)
        val notTake = dfs(wpos + 1, tpos)
        val mul = (currFreq * take) % m
        val res = (mul + notTake) % m
        cache[wpos][tpos] = res
        return res
    }
    return dfs(0, 0).toInt()
}

```

[blog post](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/solutions/3422184/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-16042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/182
#### Intuition
Consider an example: `bbc aaa ccc, target = ac`. We have 5 ways to form the `ac`:

```

// bbc aaa ccc   ac
//     a    c
//     a     c
//   c a
//      a    c
//   c  a

```

Looking at this, we deduce, that only count of every character at every position matter.

```

// 0 -> 1b 1a 1c
// 1 -> 1b 1a 1c
// 2 ->    1a 2c

```

To form `ac` we can start from position `0` or from `1`. If we start at `0`, we have one `c` at 1 plus two `c` at 2. And if we start at `1` we have two `c` at 3.
$$DP_{i,j} = Freq * DP_{i + 1, j + 1} + DP_{i + 1, j}$$

#### Approach
* precompute the `freq` array - count of each character at each position
* use an `Array` for faster cache
* use `long` to avoid overflow
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

