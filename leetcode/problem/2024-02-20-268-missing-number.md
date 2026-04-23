---
layout: leetcode-entry
title: "268. Missing Number"
permalink: "/leetcode/problem/2024-02-20-268-missing-number/"
leetcode_ui: true
entry_slug: "2024-02-20-268-missing-number"
---

[268. Missing Number](https://leetcode.com/problems/missing-number/description/) easy
[blog post](https://leetcode.com/problems/missing-number/solutions/4755419/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20022024-268-missing-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UBDYS1bz7yY)
![image.png](/assets/leetcode_daily_images/fd66aa9f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/513

#### Problem TLDR

Missing in [0..n] number.

#### Intuition

There are several ways to find it:
* subtracting sums
* doing xor
* computing sum with a math `n * (n + 1) / 2`

#### Approach

Write what is easier for you, then learn the other solutions. Xor especially.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun missingNumber(nums: IntArray): Int =
    (1..nums.size).sum() - nums.sum()

```
```rust

  pub fn missing_number(nums: Vec<i32>) -> i32 {
    nums.iter().enumerate().map(|(i, n)| i as i32 + 1 - n).sum()
  }

```

