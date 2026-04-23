---
layout: leetcode-entry
title: "1431. Kids With the Greatest Number of Candies"
permalink: "/leetcode/problem/2023-04-17-1431-kids-with-the-greatest-number-of-candies/"
leetcode_ui: true
entry_slug: "2023-04-17-1431-kids-with-the-greatest-number-of-candies"
---

[1431. Kids With the Greatest Number of Candies](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/) easy

```kotlin

fun kidsWithCandies(candies: IntArray, extraCandies: Int): List<Boolean> =
    candies.max()?.let { max ->
        candies.map { it + extraCandies >= max}
    } ?: listOf()

```

[blog post](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/solutions/3425529/kotlin-idiomatic/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-17042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/183
#### Intuition
We can just find the maximum and then try to add extra to every kid and check
#### Approach
Let's write the code
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

