---
layout: leetcode-entry
title: "1266. Minimum Time Visiting All Points"
permalink: "/leetcode/problem/2023-12-03-1266-minimum-time-visiting-all-points/"
leetcode_ui: true
entry_slug: "2023-12-03-1266-minimum-time-visiting-all-points"
---

[1266. Minimum Time Visiting All Points](https://leetcode.com/problems/minimum-time-visiting-all-points/description/) easy
[blog post](https://leetcode.com/problems/minimum-time-visiting-all-points/solutions/4356193/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03122023-1266-minimum-time-visiting?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/VjqIpuHXCF4)
![image.png](/assets/leetcode_daily_images/fe5e6e4b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/426

#### Problem TLDR

Path coordinates distance in XY plane

#### Intuition

For each pair of points lets compute diagonal distance and the remainder: `time = diag + remainder`. Given that `remainder = max(dx, dy) - diag`, we derive the formula.

#### Approach

Let's use some Kotlin's API:
* [asSequence](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/as-sequence.html)
* [windowed](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/windowed.html)
* [sumBy](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.sequences/sum-by.html)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minTimeToVisitAllPoints(points: Array<IntArray>): Int =
    points.asSequence().windowed(2).sumBy { (from, to) ->
      max(abs(to[0] - from[0]), abs(to[1] - from[1]))
    }

```

