---
layout: leetcode-entry
title: "137. Single Number II"
permalink: "/leetcode/problem/2023-07-04-137-single-number-ii/"
leetcode_ui: true
entry_slug: "2023-07-04-137-single-number-ii"
---

[137. Single Number II](https://leetcode.com/problems/single-number-ii/solutions/) medium
[blog post](https://leetcode.com/problems/single-number-ii/solutions/3715279/kotlin-o-32n/)
[substack](https://dmitriisamoilenko.substack.com/p/4072023-137-single-number-ii?sd=pf)
![image.png](/assets/leetcode_daily_images/1a9d34fb.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/265
#### Proble TLDR
Single number in an array of tripples
#### Intuition
One simple approach it to count bits at each position.
Result will have a `1` when `count % 3 != 0`.

#### Approach
Let's use fold.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun singleNumber(nums: IntArray): Int =
//110
//110
//110
//001
//001
//001
//010
//010
//010
//100
//463
(0..31).fold(0) { res, bit ->
    res or ((nums.count { 0 != it and (1 shl bit) } % 3) shl bit)
}

```

