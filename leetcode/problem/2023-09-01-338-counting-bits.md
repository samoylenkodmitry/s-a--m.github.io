---
layout: leetcode-entry
title: "338. Counting Bits"
permalink: "/leetcode/problem/2023-09-01-338-counting-bits/"
leetcode_ui: true
entry_slug: "2023-09-01-338-counting-bits"
---

[338. Counting Bits](https://leetcode.com/problems/counting-bits/description/) easy
[blog post](https://leetcode.com/problems/counting-bits/solutions/3986528/kotlin-tabulation/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/1092023-338-counting-bits?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/e3a834c6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/326

#### Problem TLDR

Array of bits count for numbers `0..n`

#### Intuition

There is a tabulation technique used for caching bits count answer in O(1): for number `xxxx0` bits count is `count(xxxx) + 0`, but for number `xxxx1` bits count is `count(xxxx) + 1`. Now, to make a switch `xxxx1 -> xxxx` simple divide by 2. Result can be cached.

#### Approach

We can use DFS + memo, but bottom-up also simple. Result is a DP array itself: `DP[number] = bits_count(number)`. The last bit can be checked by `%` operation, but `and` also works.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countBits(n: Int) = IntArray(n + 1).apply {
        for (i in 0..n)
          this[i] = this[i / 2] + (i and 1)
      }

```

