---
layout: leetcode-entry
title: "879. Profitable Schemes"
permalink: "/leetcode/problem/2023-04-21-879-profitable-schemes/"
leetcode_ui: true
entry_slug: "2023-04-21-879-profitable-schemes"
---

[879. Profitable Schemes](https://leetcode.com/problems/profitable-schemes/description/) hard

```kotlin

fun profitableSchemes(n: Int, minProfit: Int, group: IntArray, profit: IntArray): Int {
    val cache = Array(group.size) { Array(n + 1) { Array(minProfit + 1) { -1 } } }
    fun dfs(curr: Int, guys: Int, cashIn: Int): Int {
        if (guys < 0) return 0
        val cash = minOf(cashIn, minProfit)
        if (curr == group.size) return if (cash == minProfit) 1 else 0
        with(cache[curr][guys][cash]) { if (this != -1) return@dfs this }
        val notTake = dfs(curr + 1, guys, cash)
        val take = dfs(curr + 1, guys - group[curr], cash + profit[curr])
        val res = (notTake + take) % 1_000_000_007
        cache[curr][guys][cash] = res
        return res
    }
    return dfs(0, n, 0)
}

```

[blog post](https://leetcode.com/problems/profitable-schemes/solutions/3439827/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-21042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/187
#### Intuition
For every new item, `j` we can decide to take it or not take it. Given the inputs of how many `guys` we have and how much `cash` already earned, the result is always the same: $$count_j = notTake_j(cash, guys) + take_j(cash + profit[j], guys - group[j])$$

#### Approach
Do DFS and cache result in an array.
#### Complexity
- Time complexity:
$$O(n^3)$$
- Space complexity:
$$O(n^3)$$

