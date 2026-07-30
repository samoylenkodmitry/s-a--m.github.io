---
layout: leetcode-entry
title: "3014. Minimum Number of Pushes to Type Word I"
permalink: "/leetcode/problem/2026-07-30-3014-minimum-number-of-pushes-to-type-word-i/"
leetcode_ui: true
entry_slug: "2026-07-30-3014-minimum-number-of-pushes-to-type-word-i"
---

[3014. Minimum Number of Pushes to Type Word I](https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-i/solutions/8429987/kotlin-rust-by-samoylenkodmitry-agbj/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30072026-3014-minimum-number-of-pushes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/lbPpkHZS2UQ)

https://dmitrysamoylenko.com/leetcode/

![30.07.2026.webp](/assets/leetcode_daily_images/30.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1436

#### Problem TLDR

Minimum num-phone presses

#### Intuition

first eight take one click, second eight takes 2 clicks and so on

#### Approach

* and from thsi derived the math 8(1+2+..len/8) = 8 * (len/8 * (len/8 + 1)/2); plus the leftovers len%8*(len/8 + 1)

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minimumPushes(w: String) =
    w.indices.sumOf { it/8+1 }
```
```rust
    pub fn minimum_pushes(w: String) -> i32 {
        let k = w.len()/8; ((4*k+w.len()%8)*(k+1)) as _
    }
```

