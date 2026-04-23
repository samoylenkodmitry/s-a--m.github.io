---
layout: leetcode-entry
title: "2444. Count Subarrays With Fixed Bounds"
permalink: "/leetcode/problem/2024-03-31-2444-count-subarrays-with-fixed-bounds/"
leetcode_ui: true
entry_slug: "2024-03-31-2444-count-subarrays-with-fixed-bounds"
---

[2444. Count Subarrays With Fixed Bounds](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/) hard
[blog post](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/solutions/4951301/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31032024-2444-count-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YS7-vEXa6u4)
![2024-03-31_12-25.webp](/assets/leetcode_daily_images/1c7fd863.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/555

#### Problem TLDR

Count subarrays of range `minK..maxK` #hard

#### Intuition

`“all hope abandon ye who enter here”`
I've failed this question the second time (first was 1 year ago), and still find it very clever.

Consider the `safe` space as `min(a, b)..max(a,b)` where `a` is the last index of `minK` and `b` is the last index of `maxK`. We will remove suffix of `0..j` where `j` is a last out of range `minK..maxK`.

Let's examine the trick:

```j

  // 1 3 5 2 7 5      1..5
  //j
  //a
  //b
  // i
  // a
  //   i
  //     i
  //     b           +1 = min(a, b) - j = (0 - (-1))
  //       i         +1 = ...same...

```

another example:

```j

  // 0 1 2 3 4 5 6
  // 7 5 2 2 5 5 1
  //j      .
  //i      .
  //a
  //b
  // i     .
  // j     .
  //   i   .
  //   b   .
  //     i .
  //     a .         +1
  //       i
  //       a         +1
  //         i
  //         b       +3 = 3 - 0
  //           i
  //           b     +3

```

The interesting part happen at the index `i = 4`: it will update the `min(a, b)`, making it `a = 3`.

Basically, every subarray starting between `j..(min(a, b))` and ending at `i` will have minK and maxK, as `min(a,b)..max(a,b)` will have them.

#### Approach

Try to solve it yourself first.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    var res = 0L; var a = -1; var j = -1; var b = -1
    for ((i, n) in nums.withIndex()) {
      if (n == minK) a = i
      if (n == maxK) b = i
      if (n !in minK..maxK) j = i
      res += max(0, min(a, b) - j)
    }
    return res
  }

```
```rust

  pub fn count_subarrays(nums: Vec<i32>, min_k: i32, max_k: i32) -> i64 {
    let (mut res, mut a, mut b, mut j) = (0, -1, -1, -1);
    for (i, &n) in nums.iter().enumerate() {
      if n == min_k { a = i as i64 }
      if n == max_k { b = i as i64 }
      if n < min_k || n > max_k { j = i as i64 }
      res += (a.min(b) - j).max(0)
    }
    res
  }

```

