---
layout: leetcode-entry
title: "646. Maximum Length of Pair Chain"
permalink: "/leetcode/problem/2023-08-26-646-maximum-length-of-pair-chain/"
leetcode_ui: true
entry_slug: "2023-08-26-646-maximum-length-of-pair-chain"
---

[646. Maximum Length of Pair Chain](https://leetcode.com/problems/maximum-length-of-pair-chain/description/) medium
[blog post](https://leetcode.com/problems/maximum-length-of-pair-chain/solutions/3960859/kotlin-line-sweep/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26082023-646-maximum-length-of-pair?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/38e17490.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/320

#### Problem TLDR

Max count non-overlaping intervals

#### Intuition

The naive Dynamic Programming n^2 solution works, in a DFS choose between taking or skipping the pair, and cache by `pos` and `prev`.

Another solution, is just a line sweep algorithm: consider all `ends` of the intervals in increasing order, skipping the overlapping ones. It will be optimal, as there are no overlapping intervals past the `end`.

#### Approach

Sort and use the `border` variable, that changes when `from > border`.

#### Complexity

- Time complexity:
$$O(nlog(n))$$, for sorting

- Space complexity:
$$O(n)$$, for the sorted array

#### Code

```kotlin

    fun findLongestChain(pairs: Array<IntArray>): Int {
      var border = Int.MIN_VALUE
      return pairs.sortedWith(compareBy({ it[1] }))
      .count { (from, to) ->
        (from > border).also { if (it) border = to }
      }
    }

```

