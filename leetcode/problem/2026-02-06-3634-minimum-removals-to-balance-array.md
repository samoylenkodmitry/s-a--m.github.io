---
layout: leetcode-entry
title: "3634. Minimum Removals to Balance Array"
permalink: "/leetcode/problem/2026-02-06-3634-minimum-removals-to-balance-array/"
leetcode_ui: true
entry_slug: "2026-02-06-3634-minimum-removals-to-balance-array"
---

[3634. Minimum Removals to Balance Array](https://leetcode.com/problems/minimum-removals-to-balance-array/description) medium
[blog post](https://leetcode.com/problems/minimum-removals-to-balance-array/solutions/7557235/kotlin-rust-by-samoylenkodmitry-40fo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06022026-3634-minimum-removals-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IBSUuFzMoAA)

![388e4676-09f0-49fe-b476-f6dcf405a4a9 (1).webp](/assets/leetcode_daily_images/a76b7a6e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1260

#### Problem TLDR

Min removals to make min*k not less than max #medium #sliding_window

#### Intuition

```j
    // 12 18      k=2  wrong answer
    //                 why 0 instead of 1 ?
    //        ok, read description wrong
    //       it k times not k diff
    // another case is int overflow
    // 1 10 10 10 10 20
```

Invert the problem: maximum window that we want to keep is the answer.

#### Approach

* lazy sliding window: we don't have to shrink window, only care about expanding it
* use longs

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 41ms
    fun minRemoval(n: IntArray, k: Int) = n.run {
        sort(); var w = 0; count { 1L*it > 1L*k*n[w] && ++w > 0 }
    }
```
```rust
// 9ms
    pub fn min_removal(mut n: Vec<i32>, k: i32) -> i32 {
        n.sort(); let mut w = 0;
        for &x in &n { w += (x as i64 > n[w]as i64*k as i64) as usize}; w as i32
    }
```

