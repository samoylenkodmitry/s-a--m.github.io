---
layout: leetcode-entry
title: "238. Product of Array Except Self"
permalink: "/leetcode/problem/2024-03-15-238-product-of-array-except-self/"
leetcode_ui: true
entry_slug: "2024-03-15-238-product-of-array-except-self"
---

[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/) medium
[blog post](https://leetcode.com/problems/product-of-array-except-self/solutions/4877801/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15032024-238-product-of-array-except?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/P5ztPV_8dj8)
![2024-03-15_08-47.jpg](/assets/leetcode_daily_images/950ae61a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/539

#### Problem TLDR

Array of suffix-prefix products #medium

#### Intuition

Observe an example:
```j

    // 1 2 3 4
    // * 2*3*4
    // 1 * 3*4
    // 1*2 * 4
    // 1*2*3 *

```
As we can't use `/` operation, let's precompute suffix and prefix products.

#### Approach

Then we can think about the space & time optimizations.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun productExceptSelf(nums: IntArray): IntArray {
    val suf = nums.clone()
    for (i in nums.lastIndex - 1 downTo 0) suf[i] *= suf[i + 1]
    var prev = 1
    return IntArray(nums.size) { i ->
      prev * suf.getOrElse(i + 1) { 1 }.also { prev *= nums[i] }
    }
  }

```
```rust

  pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len(); let (mut res, mut p) = (vec![1; n], 1);
    for i in 1..n { res[i] = nums[i - 1] * res[i - 1] }
    for i in (0..n).rev() { res[i] *= p; p *= nums[i] }; res
  }

```

