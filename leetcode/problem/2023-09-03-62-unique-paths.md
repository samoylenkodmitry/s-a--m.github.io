---
layout: leetcode-entry
title: "62. Unique Paths"
permalink: "/leetcode/problem/2023-09-03-62-unique-paths/"
leetcode_ui: true
entry_slug: "2023-09-03-62-unique-paths"
---

[62. Unique Paths](https://leetcode.com/problems/unique-paths/description/) medium
[blog post](https://leetcode.com/problems/unique-paths/solutions/3994767/kotlin-just-sum/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3092023-62-unique-paths?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/9ade40cc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/328

#### Problem TLDR

Unique paths count, moving `right-down` from `top-left` to `bottom-right`

#### Intuition

On each cell, the number of paths is a sum of direct `up` number and direct `left` number.

#### Approach

Use single row array, as only previous up row is relevant

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin

    fun uniquePaths(m: Int, n: Int): Int {
      val row = IntArray(n) { 1 }
      for (y in 1..<m)
        for (x in 1..<n)
          row[x] += row[x - 1]
      return row.last()
    }

```

