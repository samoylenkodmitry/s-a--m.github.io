---
layout: leetcode-entry
title: "2405. Optimal Partition of String"
permalink: "/leetcode/problem/2023-04-04-2405-optimal-partition-of-string/"
leetcode_ui: true
entry_slug: "2023-04-04-2405-optimal-partition-of-string"
---

[2405. Optimal Partition of String](https://leetcode.com/problems/optimal-partition-of-string/description/) medium

[blog post](https://leetcode.com/problems/optimal-partition-of-string/solutions/3377265/kotlin-bitmask/)

```kotlin

    var mask = 0
    fun partitionString(s: String): Int = 1 + s.count {
        val bit = 1 shl (it.toInt() - 'a'.toInt())
        (mask and bit != 0).also {
            if (it) mask = 0
            mask = mask or bit
        }
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/170
#### Intuition
Expand all the intervals until they met a duplicate character. This will be the optimal solution, as the minimum of the intervals correlates with the maximum of each interval length.
#### Approach
* use `hashset`, `[26]` array or simple `32-bit` mask to store visited flags for character
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

