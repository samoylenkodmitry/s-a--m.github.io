---
layout: leetcode-entry
title: "3904. Smallest Stable Index II"
permalink: "/leetcode/problem/2026-09-05-3904-smallest-stable-index-ii/"
leetcode_ui: true
entry_slug: "2026-09-05-3904-smallest-stable-index-ii"
---

[3904. Smallest Stable Index II](https://leetcode.com/problems/smallest-stable-index-ii/solutions/8503249/kotlin-rust-by-samoylenkodmitry-obas/) medium
[substack](https://dmitriisamoilenko.substack.com/p/05092026-3904-smallest-stable-index?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LUnxr_d12FI)

https://dmitrysamoylenko.com/leetcode/

![05.09.2026.webp](/assets/leetcode_daily_images/05.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1473

#### Problem TLDR

Index of max suffix - min prefix not less than k

#### Intuition

Precompute a running minimum suffix.

#### Approach

* can be a single expression

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun firstStableIndex(n: IntArray, k: Int) = n.runningReduce(::maxOf)
        .zip(n.reversed().runningReduce(::minOf).reversed(), Int::minus)
        .indexOfFirst { it <= k }
```
```rust
    pub fn first_stable_index(n: Vec<i32>, k: i32) -> i32 {
        let (mut m, mut x) = (n.clone(), i32::MIN);
        for i in (0..n.len() - 1).rev() { m[i] = m[i].min(m[i + 1]) }
        n.iter().zip(m).position(|(&a, b)| { x = x.max(a); x - b <= k }).map_or(-1, |i| i as _)
    }
```

