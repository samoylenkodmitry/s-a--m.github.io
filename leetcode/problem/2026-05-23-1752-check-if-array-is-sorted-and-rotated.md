---
layout: leetcode-entry
title: "1752. Check if Array Is Sorted and Rotated"
permalink: "/leetcode/problem/2026-05-23-1752-check-if-array-is-sorted-and-rotated/"
leetcode_ui: true
entry_slug: "2026-05-23-1752-check-if-array-is-sorted-and-rotated"
---

[1752. Check if Array Is Sorted and Rotated](https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/solutions/8287926/kotlin-rust-by-samoylenkodmitry-0lwi/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23052026-1752-check-if-array-is-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xdsd1Wm-TsM)

https://dmitrysamoylenko.com/leetcode/

![23.05.2026.webp](/assets/leetcode_daily_images/23.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1368

#### Problem TLDR

Is array shifted and sorted?

#### Intuition

* brute-force is accepted for 100 elements
* optimal way: count unordered elements

#### Approach

* the shortest Rust is n^2
* Rust itetools has circular_tuple_windows

#### Complexity

- Time complexity:
$$O(n|n^2)$$

- Space complexity:
$$O(1|n)$$

#### Code

```kotlin
    fun check(n: IntArray) =
    n.indices.count { n[it]>n[(it+1)%n.size]} < 2
```
```rust
    pub fn check(n: Vec<i32>) -> bool {
        n.repeat(2).windows(n.len()).any(|w|w.is_sorted())
    }
```

