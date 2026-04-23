---
layout: leetcode-entry
title: "896. Monotonic Array"
permalink: "/leetcode/problem/2023-09-29-896-monotonic-array/"
leetcode_ui: true
entry_slug: "2023-09-29-896-monotonic-array"
---

[896. Monotonic Array](https://leetcode.com/problems/monotonic-array/description/) easy
[blog post](https://leetcode.com/problems/monotonic-array/solutions/4103588/kotlin-single-pass/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29092023-896-monotonic-array?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/faf96491.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/354

#### Problem TLDR

Is array monotonic

#### Intuition

Let's compute the diffs, then array is monotonic if all the diffs have the same sign.

#### Approach

Let's use Kotlin's API:
* asSequence - to avoid creating a collection
* map
* filter
* windowed - scans array by `x` sized sliding window

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun isMonotonic(nums: IntArray) =
      nums.asSequence().windowed(2)
      .map { it[0] - it[1] }
      .filter { it != 0 }
      .windowed(2)
      .all { it[0] > 0 == it[1] > 0 }

```

