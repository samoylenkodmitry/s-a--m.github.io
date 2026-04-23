---
layout: leetcode-entry
title: "1326. Minimum Number of Taps to Open to Water a Garden"
permalink: "/leetcode/problem/2023-08-31-1326-minimum-number-of-taps-to-open-to-water-a-garden/"
leetcode_ui: true
entry_slug: "2023-08-31-1326-minimum-number-of-taps-to-open-to-water-a-garden"
---

[1326. Minimum Number of Taps to Open to Water a Garden](https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/description/) hard
[blog post](https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/solutions/3983030/kotlin-greedily-fill-intervals/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31082023-1326-minimum-number-of-taps?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/d435077c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/325

#### Problem TLDR

Fill all space between `0..n` using minimum intervals

#### Intuition

We need to fill space between points, so skip all zero intervals. Next, sort intervals and scan them greedily. Consider space between `lo..hi` as filled. If `from > lo` we must open another water source. However, there are possible good candidates before, if their `to > hi`.
```
      //     0 1 2 3 4 5 6 7 8 9
      //     0 5 0 3 3 3 1 4 0 4
      // 1 5 *************
      //     ^           ^
      //     lo          hi
      // 3 3 *************
      // 4 3   *************
      //       ^         . ^
      //       from      . to
      //                 *** opened++
      //                 ^ ^
      //                lo hi
      // 5 3     *************
      //                     ^ hi
      // 7 4       *************
      //                       ^ hi finish
      // 6 1           *****
      // 9 4           *********
```

#### Approach

Look at others solutions and steal the implementation

#### Complexity
- Time complexity:
$$O(nlog(n))$$, for sorting

- Space complexity:
$$O(n)$$, to store the intervals

#### Code

```kotlin

    fun minTaps(n: Int, ranges: IntArray): Int {
      var opened = 0
      var lo = -1
      var hi = 0
      ranges.mapIndexed { i, v -> maxOf(0, i - v) to minOf(i + v, n) }
        .filter { it.first != it.second }
        .sortedBy { (from, _) -> from }
        .onEach { (from, to) ->
          if (from <= lo) hi = maxOf(hi, to)
          else if (from <= hi) {
            lo = hi
            hi = to
            opened++
          }
          if (hi == n) return opened
        }
      return -1
    }

```

