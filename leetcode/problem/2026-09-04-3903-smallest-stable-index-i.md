---
layout: leetcode-entry
title: "3903. Smallest Stable Index I"
permalink: "/leetcode/problem/2026-09-04-3903-smallest-stable-index-i/"
leetcode_ui: true
entry_slug: "2026-09-04-3903-smallest-stable-index-i"
---

[3903. Smallest Stable Index I](https://leetcode.com/problems/smallest-stable-index-i/solutions/8500950/kotlin-rust-by-samoylenkodmitry-84vk/) easy
[substack](https://dmitriisamoilenko.substack.com/p/04092026-3903-smallest-stable-index?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vebJdU38ntM)

https://dmitrysamoylenko.com/leetcode/

![04.09.2026.webp](/assets/leetcode_daily_images/04.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1472

#### Problem TLDR

Index of max suffix - min prefix not less than k

#### Intuition

Small problem, brute force is accepted.

#### Approach

* don't forget to inlcude the current position

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun firstStableIndex(n: IntArray, k: Int) =
    n.indices.find{n.take(it+1).max()-n.drop(it).min()<=k}?:-1
```
```rust
    pub fn first_stable_index(n: Vec<i32>, k: i32) -> i32 {
        (0..n.len()).find(|&i| *n[..=i].iter().max().unwrap() - *n[i..].iter().min().unwrap() <= k).map_or(-1, |i| i as _)
    }
```

