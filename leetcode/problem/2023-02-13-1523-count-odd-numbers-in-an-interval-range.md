---
layout: leetcode-entry
title: "1523. Count Odd Numbers in an Interval Range"
permalink: "/leetcode/problem/2023-02-13-1523-count-odd-numbers-in-an-interval-range/"
leetcode_ui: true
entry_slug: "2023-02-13-1523-count-odd-numbers-in-an-interval-range"
---

[1523. Count Odd Numbers in an Interval Range](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/description/) easy

[blog post](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/solutions/3179265/kotlin-o-1/)

```kotlin
    fun countOdds(low: Int, high: Int): Int {
        if (low == high) return if (low % 2 == 0) 0 else 1
        val lowOdd = low % 2 != 0
        val highOdd = high % 2 != 0
        val count = high - low + 1
        return if (lowOdd && highOdd) {
            1 + count / 2
        } else if (lowOdd || highOdd) {
            1 + (count - 1) / 2
        } else {
            1 + ((count - 2) / 2)
        }
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/117
#### Intuition
Count how many numbers in between, subtract even on the start and the end, then divide by 2.

#### Complexity
- Time complexity:
  $$O(1)$$
- Space complexity:
  $$O(1)$$

