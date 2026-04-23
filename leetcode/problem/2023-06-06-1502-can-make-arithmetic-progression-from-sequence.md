---
layout: leetcode-entry
title: "1502. Can Make Arithmetic Progression From Sequence"
permalink: "/leetcode/problem/2023-06-06-1502-can-make-arithmetic-progression-from-sequence/"
leetcode_ui: true
entry_slug: "2023-06-06-1502-can-make-arithmetic-progression-from-sequence"
---

[1502. Can Make Arithmetic Progression From Sequence](https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/description/) easy
[blog post](https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/solutions/3602840/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/06062023-1502-can-make-arithmetic?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/237
#### Problem TLDR
Is `IntArray` can be arithmetic progression?
#### Intuition
Sort, then use sliding window.

#### Approach
Let's write Kotlin one-liner.
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun canMakeArithmeticProgression(arr: IntArray): Boolean =
arr.sorted().windowed(2).groupBy { it[1] - it[0] }.keys.size == 1

```

