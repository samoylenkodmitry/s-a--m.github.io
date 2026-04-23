---
layout: leetcode-entry
title: "2966. Divide Array Into Arrays With Max Difference"
permalink: "/leetcode/problem/2024-02-01-2966-divide-array-into-arrays-with-max-difference/"
leetcode_ui: true
entry_slug: "2024-02-01-2966-divide-array-into-arrays-with-max-difference"
---

[2966. Divide Array Into Arrays With Max Difference](https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description) medium
[blog post](https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/solutions/4657723/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01022024-2966-divide-array-into-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mdtrtQGBqp0)
![image.png](/assets/leetcode_daily_images/34d9eaf3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/491

#### Problem TLDR

Split array into tripples with at most k difference.

#### Intuition

Sort, then just check `k` condition.

#### Approach

Let's use iterators in Kotlin and Rust:
* chunked vs chunks
* sorted() vs sort_unstable() (no sorted iterator in Rust)
* takeIf() vs ..
* all() vs any()
* .. map(), to_vec(), collect(), vec![]

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun divideArray(nums: IntArray, k: Int) = nums
    .sorted().chunked(3).toTypedArray()
    .takeIf { it.all { it[2] - it[0] <= k } } ?: arrayOf()

```
```rust

  pub fn divide_array(mut nums: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
    nums.sort_unstable();
    if nums.chunks(3).any(|c| c[2] - c[0] > k) { vec![] }
    else { nums.chunks(3).map(|c| c.to_vec()).collect() }
  }

```

