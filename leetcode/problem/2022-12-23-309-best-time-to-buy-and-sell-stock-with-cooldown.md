---
layout: leetcode-entry
title: "309. Best Time to Buy and Sell Stock with Cooldown"
permalink: "/leetcode/problem/2022-12-23-309-best-time-to-buy-and-sell-stock-with-cooldown/"
leetcode_ui: true
entry_slug: "2022-12-23-309-best-time-to-buy-and-sell-stock-with-cooldown"
---

[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/) medium

[https://t.me/leetcode_daily_unstoppable/61](https://t.me/leetcode_daily_unstoppable/61)

[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/1522780/java-0ms-from-recursion-with-memo-to-iterative-o-n-time-and-o-1-memory/)

```kotlin
    data class K(val a:Int, val b: Boolean, val c:Boolean)
    fun maxProfit(prices: IntArray): Int {
        val cache = mutableMapOf<K, Int>()
        fun dfs(pos: Int, canSell: Boolean, canBuy: Boolean): Int {
            return if (pos == prices.size) 0
                else cache.getOrPut(K(pos, canSell, canBuy), {
                    val profitSkip = dfs(pos+1, canSell, !canSell)
                    val profitSell = if (canSell) {prices[pos] + dfs(pos+1, false, false)} else 0
                    val profitBuy = if (canBuy) {-prices[pos] + dfs(pos+1, true, false)} else 0
                    maxOf(profitSkip, profitBuy, profitSell)
                })
        }
        return dfs(0, false, true)
    }

```

Progress from dfs solution to memo. DFS solution - just choose what to do in this step, go next, then compare results and peek max.

Space: O(N), Time: O(N)

