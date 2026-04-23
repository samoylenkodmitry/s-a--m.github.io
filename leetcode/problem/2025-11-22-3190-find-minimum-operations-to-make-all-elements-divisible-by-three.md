---
layout: leetcode-entry
title: "3190. Find Minimum Operations to Make All Elements Divisible by Three"
permalink: "/leetcode/problem/2025-11-22-3190-find-minimum-operations-to-make-all-elements-divisible-by-three/"
leetcode_ui: true
entry_slug: "2025-11-22-3190-find-minimum-operations-to-make-all-elements-divisible-by-three"
---

[3190. Find Minimum Operations to Make All Elements Divisible by Three](https://leetcode.com/problems/find-minimum-operations-to-make-all-elements-divisible-by-three) easy
[blog post](https://leetcode.com/problems/find-minimum-operations-to-make-all-elements-divisible-by-three/solutions/7366515/kotlin-rust-by-samoylenkodmitry-8acp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22112025-3190-find-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/fFrsEbcMsek)

![9e398533-0b61-4a05-b20f-168a4d7d0dcc (1).webp](/assets/leetcode_daily_images/e5cfe688.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1181

#### Problem TLDR

Min steps inc/dec to make %3 #easy

#### Intuition

* min(x%3, 3-x%3)

#### Approach

* it is actually always min(1,2) = 1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 1ms
    fun minimumOperations(n: IntArray) =
        n.count { it%3>0 }
```
```rust
// 0ms
    pub fn minimum_operations(n: Vec<i32>) -> i32 {
        n.iter().filter(|&x| x%3>0).count() as _
    }
```

