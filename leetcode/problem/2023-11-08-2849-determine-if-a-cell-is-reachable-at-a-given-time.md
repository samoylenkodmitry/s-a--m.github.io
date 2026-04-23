---
layout: leetcode-entry
title: "2849. Determine if a Cell Is Reachable at a Given Time"
permalink: "/leetcode/problem/2023-11-08-2849-determine-if-a-cell-is-reachable-at-a-given-time/"
leetcode_ui: true
entry_slug: "2023-11-08-2849-determine-if-a-cell-is-reachable-at-a-given-time"
---

[2849. Determine if a Cell Is Reachable at a Given Time](https://leetcode.com/problems/determine-if-a-cell-is-reachable-at-a-given-time/description/) medium
[blog post](https://leetcode.com/problems/determine-if-a-cell-is-reachable-at-a-given-time/solutions/4262992/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08112023-2849-determine-if-a-cell?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/25c36adc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/397

#### Problem TLDR

Is path possible on grid `sx, sy -> fx, fy`

#### Intuition

Given the problem size, we can't use DP, as it will take O(n^2). However, we must notice, that if the shortest path is reachable, than any other path can be formed to travel at any time.

#### Approach

The shortest path will consist of only the difference between coordinates.

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun isReachableAtTime(sx: Int, sy: Int, fx: Int, fy: Int, t: Int): Boolean {
      var dx = Math.abs(fx - sx)
      var dy = Math.abs(fy - sy)
      var both = min(dx, dy)
      var other = max(dx, dy) - both
      var total = both + other
      return total <= t && (total > 0 || t != 1)
    }

```

