---
layout: leetcode-entry
title: "1351. Count Negative Numbers in a Sorted Matrix"
permalink: "/leetcode/problem/2025-12-28-1351-count-negative-numbers-in-a-sorted-matrix/"
leetcode_ui: true
entry_slug: "2025-12-28-1351-count-negative-numbers-in-a-sorted-matrix"
---

[1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/description/) easy
[blog post](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/solutions/7445448/kotlin-rust-by-samoylenkodmitry-ak5l/)
[substack](https://dmitriisamoilenko.substack.com/publish/posts/detail/182755897/share-center)
[youtube](https://youtu.be/YTLqeFL620U)

![6740ea68-029c-43e8-a7e2-f5ebb31cbb30 (1).webp](/assets/leetcode_daily_images/473a6bc2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1218

#### Problem TLDR

Count negatives in 2D sorted matrix #easy

#### Intuition

Brute force.
Improve with either:
* binary search
* n+m walk on border of negatives

#### Approach

* use list.binarySearch {..} in Kotlin or partition_point in Rust

#### Complexity

- Time complexity:
$$O(nlogm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin
// 15ms
    fun countNegatives(g: Array<IntArray>) =
        g.sumOf { g[0].size+1+it.asList().binarySearch { if (it < 0) 1 else -1 } }
```
```rust
// 0ms
    pub fn count_negatives(g: Vec<Vec<i32>>) -> i32 {
        g.iter().map(|r| { (r.len() - r.partition_point(|&x| x >= 0)) as i32 }).sum()
    }
```

