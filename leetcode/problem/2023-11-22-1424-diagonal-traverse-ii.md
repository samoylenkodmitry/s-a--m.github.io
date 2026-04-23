---
layout: leetcode-entry
title: "1424. Diagonal Traverse II"
permalink: "/leetcode/problem/2023-11-22-1424-diagonal-traverse-ii/"
leetcode_ui: true
entry_slug: "2023-11-22-1424-diagonal-traverse-ii"
---

[1424. Diagonal Traverse II](https://leetcode.com/problems/diagonal-traverse-ii/description/) medium
[blog post](https://leetcode.com/problems/diagonal-traverse-ii/solutions/4315814/kotlin-priorityqueue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22112023-1424-diagonal-traverse-ii?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/70a4f186.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/413

#### Problem TLDR

Diagonal 2D matrix order with prunes

#### Intuition

The naive solution is to adjust the pointers `x` and `y`. However, that will cost O(max(x)*max(y)) and give TLE.

Let's just sort indices pairs `(x y)` and take them one by one.

#### Approach

Use some Kotlin's features:
* with
* let
* indices
* compareBy({ one }, { two })

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findDiagonalOrder(nums: List<List<Int>>): IntArray =
    with(PriorityQueue<Pair<Int, Int>>(compareBy(
      { it.first + it.second }, { it.first }, { it.second }
    ))) {
    for (y in nums.indices)
      for (x in nums[y].indices) add(x to y)
    IntArray(size) { poll().let { (x, y) -> nums[y][x]} }
  }

```

