---
layout: leetcode-entry
title: "1822. Sign of the Product of an Array"
permalink: "/leetcode/problem/2023-05-02-1822-sign-of-the-product-of-an-array/"
leetcode_ui: true
entry_slug: "2023-05-02-1822-sign-of-the-product-of-an-array"
---

[1822. Sign of the Product of an Array](https://leetcode.com/problems/sign-of-the-product-of-an-array/description/) easy

```kotlin

fun arraySign(nums: IntArray): Int = nums.fold(1) { r, t -> if (t == 0) 0 else r * (t / Math.abs(t)) }

```

[blog post](https://leetcode.com/problems/sign-of-the-product-of-an-array/solutions/3475973/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-2052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/199
#### Intuition
Do what is asked, but avoid overflow.

#### Approach
There is an `sign` function in kotlin, but leetcode.com doesn't support it yet.
We can use `fold`.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

