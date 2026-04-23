---
layout: leetcode-entry
title: "2946. Matrix Similarity After Cyclic Shifts"
permalink: "/leetcode/problem/2026-03-27-2946-matrix-similarity-after-cyclic-shifts/"
leetcode_ui: true
entry_slug: "2026-03-27-2946-matrix-similarity-after-cyclic-shifts"
---

[2946. Matrix Similarity After Cyclic Shifts]() easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27032026-2946-matrix-similarity-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/L8wxjcj09WE)

![27032026.webp](/assets/leetcode_daily_images/27.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1310

#### Problem TLDR

Shift rows k times left and right #easy #matrxi

#### Intuition

Brute force is accepted.

#### Approach

* we can shift by k instead of by 1 k times
* rows must be periodic or size == k, so left shift is the same as right shift
* just check each row, don't have to create a matrix

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 15ms
    fun areSimilar(m: Array<IntArray>, k: Int) =
    m.all { r -> r.indices.all { r[it] == r[(it+k)%r.size] }}
```
```rust
// 0ms
    pub fn are_similar(m: Vec<Vec<i32>>, k: i32) -> bool {
        m.iter().all(|r| (0..r.len()).all(|i| r[i] == r[(i + k as usize) % r.len()]))
    }
```

