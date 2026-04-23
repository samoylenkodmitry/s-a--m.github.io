---
layout: leetcode-entry
title: "2840. Check if Strings Can be Made Equal With Operations II"
permalink: "/leetcode/problem/2026-03-30-2840-check-if-strings-can-be-made-equal-with-operations-ii/"
leetcode_ui: true
entry_slug: "2026-03-30-2840-check-if-strings-can-be-made-equal-with-operations-ii"
---

[2840. Check if Strings Can be Made Equal With Operations II](https://open.substack.com/pub/dmitriisamoilenko/p/30032026-2840-check-if-strings-can?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium

[youtube](https://youtu.be/QTB-Cqu_S-w)

![30.03.2026.webp](/assets/leetcode_daily_images/30.03.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1313

#### Problem TLDR

Strings equal at odd-even positions #medium #hash

#### Intuition

1. compare frequencies
2. compare hashes

#### Approach

* as well as it is green its all fine! (for golf)
* sum((c|parity)^4) is the perfect hash

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 49ms
    fun checkStrings(a: String, b: String) = a.indices.sumOf { i ->
        (a[i]-' '+i%2*32.0).pow(3) - (b[i]-' '+i%2*32.0).pow(3)
    } == 0.0
```
```rust
// 0ms
    pub fn check_strings(a: String, b: String) -> bool {
        let h = |s: &[u8]| (0..s.len()).map(|i|
            1<<(s[i] as usize & 31 | i%2*7) ).sum::<usize>();
        h(a.as_bytes()) == h(b.as_bytes())
    }
```

