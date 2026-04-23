---
layout: leetcode-entry
title: "121. Best Time to Buy and Sell Stock"
permalink: "/leetcode/problem/2023-02-25-121-best-time-to-buy-and-sell-stock/"
leetcode_ui: true
entry_slug: "2023-02-25-121-best-time-to-buy-and-sell-stock"
---

[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/) easy

[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/solutions/3227923/kotlin-min-max/)

```kotlin

fun maxProfit(prices: IntArray): Int {
    var min = prices[0]
    var profit = 0
    prices.forEach {
        if (it < min) min = it
        profit = maxOf(profit, it - min)
    }
    return profit
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/129
#### Intuition
Max profit will be the difference between `max` and `min`. One thing to note, the `max` must follow after the `min`.

#### Approach
* we can just use current value as a `max` candidate instead of managing the `max` variable.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

