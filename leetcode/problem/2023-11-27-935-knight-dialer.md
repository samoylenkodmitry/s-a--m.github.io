---
layout: leetcode-entry
title: "935. Knight Dialer"
permalink: "/leetcode/problem/2023-11-27-935-knight-dialer/"
leetcode_ui: true
entry_slug: "2023-11-27-935-knight-dialer"
---

[935. Knight Dialer](https://leetcode.com/problems/knight-dialer/description/) medium
[blog post](https://leetcode.com/problems/knight-dialer/solutions/4334170/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27112023-935-knight-dialer?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/cr2WnuMLRsQ)
![image.png](/assets/leetcode_daily_images/dc7374dc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/418

#### Problem TLDR

Count of dialer `n`-length numbers formed by pressing in a chess Knight's moves

#### Intuition

We can search with Depth-First Search for every position and count every path that has `n` digits in it.
The result will only depend on a previous number and count of the remaining moves, so can be cached.

#### Approach

Let's write a separate `paths` map: current digit to next possible.

#### Complexity

- Time complexity:
$$O(n)$$, `10` digits is a constant value

- Space complexity:
$$O(n)$$

#### Code

```kotlin
  val dp = mutableMapOf<Pair<Int, Int>, Int>()
  val paths = mapOf(
    -1 to (0..9).toList(),
    0 to listOf(4, 6),
    1 to listOf(6, 8),
    2 to listOf(7, 9),
    3 to listOf(4, 8),
    4 to listOf(3, 9, 0),
    5 to listOf(),
    6 to listOf(1, 7, 0),
    7 to listOf(2, 6),
    8 to listOf(1, 3),
    9 to listOf(2, 4))
  fun knightDialer(pos: Int, prev: Int = -1): Int =
    if (pos == 0) 1 else dp.getOrPut(pos to prev) {
      paths[prev]!!.map { knightDialer(pos - 1, it) }
      .fold(0) { r, t -> (r + t) % 1_000_000_007 }
    }

```

