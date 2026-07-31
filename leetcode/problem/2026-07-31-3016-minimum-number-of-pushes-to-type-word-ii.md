---
layout: leetcode-entry
title: "3016. Minimum Number of Pushes to Type Word II"
permalink: "/leetcode/problem/2026-07-31-3016-minimum-number-of-pushes-to-type-word-ii/"
leetcode_ui: true
entry_slug: "2026-07-31-3016-minimum-number-of-pushes-to-type-word-ii"
---

[3016. Minimum Number of Pushes to Type Word II](https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/solutions/8432433/kotlin-rust-by-samoylenkodmitry-oq9t/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31072026-3016-minimum-number-of-pushes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lbPpkHZS2UQ)

https://dmitrysamoylenko.com/leetcode/

![31.07.2026.webp](/assets/leetcode_daily_images/31.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1437

#### Problem TLDR

Minimum old-phone keypresses

#### Intuition

Sort letters by frequencies. Put most frequen on first columns, then on second, third, etc.

#### Approach

* use chunks or i/8 + 1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minimumPushes(w: String) = w.groupBy{it}.values
    .sortedBy{-it.size}.mapIndexed{i,t->(i/8+1)*t.size}.sum()
```
```rust
    pub fn minimum_pushes(w: String) -> i32 {
        w.bytes().counts().values().sorted().rev()
        .zip(8..).map(|(&f, i)| (f * (i / 8)) as i32).sum()
    }
```

