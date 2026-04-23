---
layout: leetcode-entry
title: "1287. Element Appearing More Than 25% In Sorted Array"
permalink: "/leetcode/problem/2023-12-11-1287-element-appearing-more-than-25-in-sorted-array/"
leetcode_ui: true
entry_slug: "2023-12-11-1287-element-appearing-more-than-25-in-sorted-array"
---

[1287. Element Appearing More Than 25% In Sorted Array](https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/description/) easy
[blog post](https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/solutions/4389153/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11122023-1287-element-appearing-more?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/2856bc33.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/435

#### Problem TLDR

Most frequent element

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1)

#### Code

```kotlin

  fun findSpecialInteger(arr: IntArray): Int =
    arr.groupBy { it }
      .maxBy { (k, v) -> v.size }!!
      .key

```

