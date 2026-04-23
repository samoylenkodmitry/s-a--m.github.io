---
layout: leetcode-entry
title: "713. Subarray Product Less Than K"
permalink: "/leetcode/problem/2024-03-27-713-subarray-product-less-than-k/"
leetcode_ui: true
entry_slug: "2024-03-27-713-subarray-product-less-than-k"
---

[713. Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/description/) medium
[blog post](https://leetcode.com/problems/subarray-product-less-than-k/solutions/4931440/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27032024-713-subarray-product-less?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HTLfyj_ghYs)
![2024-03-27_09-18.webp](/assets/leetcode_daily_images/c03d7ba4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/551

#### Problem TLDR

Subarrays count with product less than `k` #medium

#### Intuition

Let's try to use two pointers and move them only once:

```j
  // 10 5 2 6 1 1 1                    cnt
  // i                    10           1
  // j
  // *  j                 50 +5        3
  //    * j               (100) +2     4
  //    i                 10           5
  //    * * j             60 +6        7
  //    * * * j           60 +1        9
  //    * * * * j         60 +1        11
  //    * * * * * j       60 +1        13
  //      i * * * *       12 +1        15
  //        i * * *       6 +1         17
  //          i * *       1 +1         19
  //            i *       1 +1         21
  //              i       1 +1         23
```
As we notice, this way gives the correct answer. Expand the first pointer while  `p < k`, then shrink the second pointer.

#### Approach

Next, some tricks:
* move the right pointer once at a time
* move the second until conditions are met
* adding `(i - j)` helps to avoid moving the left pointer
* if we handle the corner cases of `k = 0` and `k = 1`, we can use some optimizations: `nums[j]` will always be less than `k` after `while` loop; and `i` will always be less than `i` in a `while` loop.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun numSubarrayProductLessThanK(nums: IntArray, k: Int): Int {
    var i = 0; var j = 0; var res = 0; var p = 1
    if (k < 2) return 0
    for (j in nums.indices) {
      p *= nums[j]
      while (p >= k) p /= nums[i++]
      res += j - i + 1
    }
    return res
  }

```
```rust

  pub fn num_subarray_product_less_than_k(nums: Vec<i32>, k: i32) -> i32 {
    if k < 2 { return 0 }
    let (mut j, mut p, mut res) = (0, 1, 0);
    for i in 0..nums.len() {
      p *= nums[i];
      while p >= k { p /= nums[j]; j += 1 }
      res += i - j + 1
    }
    res as i32
  }

```

