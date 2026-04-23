---
layout: leetcode-entry
title: "6. Zigzag Conversion"
permalink: "/leetcode/problem/2023-02-03-6-zigzag-conversion/"
leetcode_ui: true
entry_slug: "2023-02-03-6-zigzag-conversion"
---

[6. Zigzag Conversion](https://leetcode.com/problems/zigzag-conversion/description/) medium

[blog post](https://leetcode.com/problems/zigzag-conversion/solutions/3135114/kotlin-simulation/)

```kotlin
    fun convert(s: String, numRows: Int): String {
        if (numRows <= 1) return s
        // nr = 5
        //
        // 0    8       16        24
        // 1   7 9     15 17     23 25
        // 2  6  10   14   18   22   26   30
        // 3 5    11 13     19 21     27 29
        // 4       12        20        28
        //
        val indices = Array(numRows) { mutableListOf<Int>() }
        var y = 0
        var dy = 1
        for (i in 0..s.lastIndex) {
            indices[y].add(i)
            if (i > 0 && (i % (numRows - 1)) == 0) dy = -dy
            y += dy
        }
        return StringBuilder().apply {
            indices.forEach { it.forEach { append(s[it]) } }
        }.toString()
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/107
#### Intuition

```

        // nr = 5
        //
        // 0    8       16        24
        // 1   7 9     15 17     23 25
        // 2  6  10   14   18   22   26   30
        // 3 5    11 13     19 21     27 29
        // 4       12        20        28
        //

```

We can just simulate zigzag.
#### Approach
Store simulation result in a `[rowsNum][simulation indice]` - matrix, then build the result.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

