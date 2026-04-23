---
layout: leetcode-entry
title: "518. Coin Change II"
permalink: "/leetcode/problem/2023-08-11-518-coin-change-ii/"
leetcode_ui: true
entry_slug: "2023-08-11-518-coin-change-ii"
---

[518. Coin Change II](https://leetcode.com/problems/coin-change-ii/description/) medium
[blog post](https://leetcode.com/problems/coin-change-ii/solutions/3893011/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11082023-518-coin-change-ii?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/2ce10fa4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/305

#### Problem TLDR

Ways to make `amount` with array of `coins`

#### Intuition

This is a classical Dynamic Programming problem: the result is only depending on inputs – `coins` subarray and the `amount`, so can be cached.

In a Depth-First search manner, consider possibilities of `taking` a coin and `skipping` to the next.

#### Approach

* HashMap gives TLE, but an Array cache will pass

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun change(amount: Int, coins: IntArray): Int {
      val cache = Array(coins.size) { IntArray(amount + 1) { -1 } }
      fun dfs(curr: Int, left: Int): Int = if (left == 0) 1
        else if (left < 0 || curr == coins.size) 0
        else cache[curr][left].takeIf { it >= 0 } ?: {
          dfs(curr, left - coins[curr]) + dfs(curr + 1, left)
        }().also { cache[curr][left] = it }
      return dfs(0, amount)
    }

```

