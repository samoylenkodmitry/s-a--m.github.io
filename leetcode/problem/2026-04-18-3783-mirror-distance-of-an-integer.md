---
layout: leetcode-entry
title: "3783. Mirror Distance of an Integer"
permalink: "/leetcode/problem/2026-04-18-3783-mirror-distance-of-an-integer/"
leetcode_ui: true
entry_slug: "2026-04-18-3783-mirror-distance-of-an-integer"
---

[3783. Mirror Distance of an Integer](https://leetcode.com/problems/mirror-distance-of-an-integer/solutions/7976571/kotlin-rust-by-samoylenkodmitry-nu3r/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18042026-3783-mirror-distance-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MuPXg9xC9Hg)

![18.04.2026.webp](/assets/leetcode_daily_images/18.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1332

#### Problem TLDR

Diff with reversal #easy

#### Intuition

Reverse and subtract.
Can't be done in-place, because we need to know the length of number.

#### Approach

* Kotlin's shortest is strings reversal
* Rust doesn't have divmod, but can use asm that will mutate 'x' and return reminder `let d: i32; unsafe { asm!("cdq;div {0}", in(reg) 10, inout("eax") x, out("edx") d) }`

#### Complexity

- Time complexity:
$$O(lg(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 10ms
    fun mirrorDistance(n: Int) =
    abs(n - "$n".reversed().toInt())
```
```rust
// 0ms
    pub fn mirror_distance(n: i32) -> i32 {
        let (mut r, mut x) = (0, n);
        while x > 0 { r = r*10+x%10; x/=10}; (r-n).abs()
    }
```

