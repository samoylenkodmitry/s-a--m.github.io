---
layout: leetcode-entry
title: "714. Best Time to Buy and Sell Stock with Transaction Fee"
permalink: "/leetcode/problem/2023-06-22-714-best-time-to-buy-and-sell-stock-with-transaction-fee/"
leetcode_ui: true
entry_slug: "2023-06-22-714-best-time-to-buy-and-sell-stock-with-transaction-fee"
---

[714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/) medium
[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solutions/3668167/kotlin-track-money-balance/)
[substack](https://dmitriisamoilenko.substack.com/p/22062023-714-best-time-to-buy-and?sd=pf)
![image.png](/assets/leetcode_daily_images/4524f9b6.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/253
#### Problem TLDR
Max profit from buying stocks and selling them with `fee` for `prices[day]`
#### Intuition
Naive recursive or iterative Dynamic Programming solution will take $$O(n^2)$$ time if we iterate over all days for buying and for selling.
The trick here is to consider the money balances you have each day. We can track two separate money balances: for when we're buying the stock `balanceBuy` and for when we're selling `balanceSell`. Then, it is simple to greedily track balances:
* if we choose to buy, we subtract `prices[day]` from `balanceBuy`
* if we choose to sell, we add `prices[day] - fee` to `balanceSell`
* just greedily compare previous balances with choices and choose maximum balance.

#### Approach
* balances are always following each other: `buy-sell-buy-sell..`, or we can rewrite this like `currentBalance = maxOf(balanceSell, balanceBuy)` and use it for addition and subtraction.
* we can keep only the previous balances, saving space to $$O(1)$$
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxProfit(prices: IntArray, fee: Int) = prices
.fold(-prices[0] to 0) { (balanceBuy, balance), price ->
    maxOf(balanceBuy, balance - price) to maxOf(balance, balanceBuy + price - fee)
}.second

```

