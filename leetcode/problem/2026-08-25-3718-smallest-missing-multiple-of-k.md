---
layout: leetcode-entry
title: "3718. Smallest Missing Multiple of K"
permalink: "/leetcode/problem/2026-08-25-3718-smallest-missing-multiple-of-k/"
leetcode_ui: true
entry_slug: "2026-08-25-3718-smallest-missing-multiple-of-k"
---

[3718. Smallest Missing Multiple of K](https://leetcode.com/problems/smallest-missing-multiple-of-k/solutions/8481562/kotlin-rust-by-samoylenkodmitry-r9ji/) easy
[substack](https://dmitriisamoilenko.substack.com/p/25082026-3718-smallest-missing-multiple?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NvyYK0po758)

https://dmitrysamoylenko.com/leetcode/

![25.08.2026.webp](/assets/leetcode_daily_images/25.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1462

#### Problem TLDR

Smallest k-multiplier not in an array

#### Intuition

Brute-force

#### Approach

* what is the upper bound?

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun missingMultiple(n: IntArray, k: Int) =
    (k..200 step k).find { it !in n }
```
```rust
    pub fn missing_multiple(n: Vec<i32>, k: i32) -> i32 {
        (1..).find(|i| !n.contains(&(i * k))).unwrap() * k
    }
```

