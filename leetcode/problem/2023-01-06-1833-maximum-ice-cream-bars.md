---
layout: leetcode-entry
title: "1833. Maximum Ice Cream Bars"
permalink: "/leetcode/problem/2023-01-06-1833-maximum-ice-cream-bars/"
leetcode_ui: true
entry_slug: "2023-01-06-1833-maximum-ice-cream-bars"
---

[1833. Maximum Ice Cream Bars](https://leetcode.com/problems/maximum-ice-cream-bars/description/) medium

[https://t.me/leetcode_daily_unstoppable/77](https://t.me/leetcode_daily_unstoppable/77)

[blog post](https://leetcode.com/problems/maximum-ice-cream-bars/solutions/3007210/kotlin-greedy/)

```kotlin
    fun maxIceCream(costs: IntArray, coins: Int): Int {
       costs.sort()
       var coinsRemain = coins
       var iceCreamCount = 0
       for (i in 0..costs.lastIndex) {
           coinsRemain -= costs[i]
           if (coinsRemain < 0) break
           iceCreamCount++
       }
       return iceCreamCount
    }

```

The `maximum ice creams` would be if we take as many `minimum costs` as possible
Sort the `costs` array, then greedily iterate it and buy ice creams until all the coins are spent.

Space: O(1), Time: O(NlogN) (there is also O(N) solution based on count sort)

