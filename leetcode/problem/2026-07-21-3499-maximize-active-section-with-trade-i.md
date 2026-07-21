---
layout: leetcode-entry
title: "3499. Maximize Active Section with Trade I"
permalink: "/leetcode/problem/2026-07-21-3499-maximize-active-section-with-trade-i/"
leetcode_ui: true
entry_slug: "2026-07-21-3499-maximize-active-section-with-trade-i"
---

[3499. Maximize Active Section with Trade I](https://leetcode.com/problems/maximize-active-section-with-trade-i/solutions/8410725/kotlin-rust-by-samoylenkodmitry-s4nd/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21072026-3499-maximize-active-section?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jG3cmEjBY7c)

https://dmitrysamoylenko.com/leetcode/

![21.07.2026.webp](/assets/leetcode_daily_images/21.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1427

#### Problem TLDR

Max ones after replacing surrounding zeros

#### Intuition

Count max of surrounding zeros plus total ones.

#### Approach

* only 4 variables necessary, but we can count ones in the same loop, that adds variables

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxActiveSectionsAfterTrade(s: String): Int {
        var pp = 0; var p = 0; var c = 0; var ch = '1'
        return s.count {it=='1'}+s.maxOf { x ->
            if (x == ch) c++  else { pp = p; p = c; c = 1; ch = x }
            if (x == '0' && p>0 && pp>0) pp+c else 0
        }
    }
```
```rust
    pub fn max_active_sections_after_trade(s: String) -> i32 {
        let (mut pp, mut p, mut c, mut k) = (0, 0, 0, 0);
        s.bytes().filter(|&b| b == 49).count() as i32 + s.bytes().map(|b| {
            if b == k { c += 1 } else { (pp, p, c, k) = (p, c, 1, b) }
            if b == 48 && pp > 0 { pp + c } else { 0 }
        }).max().unwrap_or(0)
    }
```

