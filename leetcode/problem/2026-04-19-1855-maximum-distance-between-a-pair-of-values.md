---
layout: leetcode-entry
title: "1855. Maximum Distance Between a Pair of Values"
permalink: "/leetcode/problem/2026-04-19-1855-maximum-distance-between-a-pair-of-values/"
leetcode_ui: true
entry_slug: "2026-04-19-1855-maximum-distance-between-a-pair-of-values"
---

[1855. Maximum Distance Between a Pair of Values](https://leetcode.com/problems/maximum-distance-between-a-pair-of-values/solutions/7992229/kotlin-rust-by-samoylenkodmitry-9w7q/) medium

[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19042026-1855-maximum-distance-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7z_Q-wOUwzQ)

![19.04.2026.webp](/assets/leetcode_daily_images/19.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1333

#### Problem TLDR

Max distance a[i] not greater than b[j] #medium #sliding_window

#### Intuition

1. Sliding window: for each left pointer move the rigth as far as possible
2. Max-window: always expand the window, shrink window only when condition is broken

#### Approach

* the max-window is more clever
* binary search would also work

#### Complexity

- Time complexity:
$$O(n)$$, nlog(n) for binarysearch

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 81ms
    fun maxDistance(a: IntArray, b: IntArray) = maxOf(0, a.indices
    .maxOf { i -> -b.asList().binarySearch { if (it < a[i]) 1 else -1 }-i-2 })
```
```rust
// 3ms
    pub fn max_distance(a: Vec<i32>, b: Vec<i32>) -> i32 {
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() { if a[i] > b[j] { i += 1 }; j += 1}
        0.max(j as i32 - i as i32 - 1)
    }
```

