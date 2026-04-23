---
layout: leetcode-entry
title: "3512. Minimum Operations to Make Array Sum Divisible by K"
permalink: "/leetcode/problem/2025-11-29-3512-minimum-operations-to-make-array-sum-divisible-by-k/"
leetcode_ui: true
entry_slug: "2025-11-29-3512-minimum-operations-to-make-array-sum-divisible-by-k"
---

[3512. Minimum Operations to Make Array Sum Divisible by K](https://leetcode.com/problems/minimum-operations-to-make-array-sum-divisible-by-k/description/) easy
[blog post](https://leetcode.com/problems/minimum-operations-to-make-array-sum-divisible-by-k/solutions/7381053/kotlin-rust-by-samoylenkodmitry-vf9g/)
[substack]()
[youtube](https://youtu.be/0lz_HyqO77A)

![1e4c53de-2b51-4294-b404-135d7f781973 (1).webp](/assets/leetcode_daily_images/8143f054.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1188

#### Problem TLDR

Sum % k #easy

#### Intuition

The number of operations is the remainder of %k

#### Approach

* %k

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 11ms
    fun minOperations(n: IntArray, k: Int) =
        n.sum() % k
```
```rust
// 0ms
    pub fn min_operations(n: Vec<i32>, k: i32) -> i32 {
        n.iter().sum::<i32>() % k
    }
```

