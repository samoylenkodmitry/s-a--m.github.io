---
layout: leetcode-entry
title: "1921. Eliminate Maximum Number of Monsters"
permalink: "/leetcode/problem/2023-11-07-1921-eliminate-maximum-number-of-monsters/"
leetcode_ui: true
entry_slug: "2023-11-07-1921-eliminate-maximum-number-of-monsters"
---

[1921. Eliminate Maximum Number of Monsters](https://leetcode.com/problems/eliminate-maximum-number-of-monsters/description/) medium
[blog post](https://leetcode.com/problems/eliminate-maximum-number-of-monsters/solutions/4259171/kotlin-sort-by-arrival-time/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07112023-1921-eliminate-maximum-number?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/c064c6f3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/396

#### Problem TLDR

Count possible `1-minute` kills in a game of `dist[]` targets falling with `speed[]`

#### Intuition

Each target has it's own `arrival time_i = dist[i] / speed[i]`. We must prioritize targets by it.

#### Approach

Let's use Kotlin API:

* indices
* sortedBy
* withIndex
* takeWhile
* time becomes just a target index

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun eliminateMaximum(dist: IntArray, speed: IntArray): Int =
      dist.indices.sortedBy { dist[it].toDouble() / speed[it] }
      .withIndex()
      .takeWhile { (time, ind) -> speed[ind] * time < dist[ind] }
      .count()

```

