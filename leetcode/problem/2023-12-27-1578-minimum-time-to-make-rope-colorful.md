---
layout: leetcode-entry
title: "1578. Minimum Time to Make Rope Colorful"
permalink: "/leetcode/problem/2023-12-27-1578-minimum-time-to-make-rope-colorful/"
leetcode_ui: true
entry_slug: "2023-12-27-1578-minimum-time-to-make-rope-colorful"
---

[1578. Minimum Time to Make Rope Colorful](https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/) medium
[blog post](https://leetcode.com/problems/minimum-time-to-make-rope-colorful/solutions/4464920/kotlin-greedy-scan/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27122023-1578-minimum-time-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/JBitP1oM2Ac)
![image.png](/assets/leetcode_daily_images/213d2bcc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/452

#### Problem TLDR

Min sum of removed duplicates in array.

#### Intuition

The brute-force approach is to just consider keeping/remove every item, that can be cached in [size, 26] array.

However, there is a more optimal greedy solution: scan symbols one by one, and from each duplicate island remove the maximum of it.

#### Approach

Start from writing more verbose solution, keeping separate variables for `currentSum`, `totalSum`, and two separate conditions: if we meet a duplicate or not.
Then optimize it out.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minCost(colors: String, neededTime: IntArray): Int {
    var sum = 0
    var max = 0
    var prev = '.'
    for ((i, c) in colors.withIndex()) {
      sum += neededTime[i]
      if (prev != c) sum -= max.also { max = 0 }
      max = max(max, neededTime[i])
      prev = c
    }
    return sum - max
  }

```

