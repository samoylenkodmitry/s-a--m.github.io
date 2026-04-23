---
layout: leetcode-entry
title: "1561. Maximum Number of Coins You Can Get"
permalink: "/leetcode/problem/2023-11-24-1561-maximum-number-of-coins-you-can-get/"
leetcode_ui: true
entry_slug: "2023-11-24-1561-maximum-number-of-coins-you-can-get"
---

[1561. Maximum Number of Coins You Can Get](https://leetcode.com/problems/maximum-number-of-coins-you-can-get/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-coins-you-can-get/solutions/4323147/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24112023-1561-maximum-number-of-coins?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/wsx94MZvALk)
![image.png](/assets/leetcode_daily_images/b808afae.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/415

#### Problem TLDR

Get sum of second maxes of triples from array

#### Intuition

Observing the example:

```kotlin
  // 1 2 3 4 5 6 7 8 9
  // *             * *  8
  //   *       * *      6
  //     * * *          4
  // size = x + 2x
```
we can deduce an optimal algorithm: give bob the smallest value, and take the second largest. There are exactly `size / 3` moves total.

#### Approach

Let's write it in a functional style, using Kotlin's API:
* sorted
* drop
* chunked
* sumBy

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$, can be O(1) when sorted in-place

#### Code

```kotlin

  fun maxCoins(piles: IntArray): Int =
    piles.sorted()
      .drop(piles.size / 3)
      .chunked(2)
      .sumBy { it[0] }

```

