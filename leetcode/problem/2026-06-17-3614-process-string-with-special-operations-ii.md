---
layout: leetcode-entry
title: "3614. Process String with Special Operations II"
permalink: "/leetcode/problem/2026-06-17-3614-process-string-with-special-operations-ii/"
leetcode_ui: true
entry_slug: "2026-06-17-3614-process-string-with-special-operations-ii"
---

[3614. Process String with Special Operations II](https://leetcode.com/problems/process-string-with-special-operations-ii/solutions/8339582/kotlin-rust-by-samoylenkodmitry-266t/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17062026-3614-process-string-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vrFeFK_b5lU)

https://dmitrysamoylenko.com/leetcode/

![17.06.2026.webp](/assets/leetcode_daily_images/17.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1393

#### Problem TLDR

K-th char ina pattern-build a string, %-reverse, #-repeat, *-pop

#### Intuition

```j
    // cd%#*# k=3
    // dc
    // dcdc
    // dcd
    // dcddcd  len=6
    // ...k
    // #       len=3 k=0
    // dcd
    // k
    // *       len=4
    // dcd*
    // #       len=2 k=0
    // dc
    //
```

Simulate the rules to find the length.
Reverse operations: * adds to length, # halfs length and trims K-length, % reverses k = len-1-k

#### Approach

* corner cases: k >= length, max(0, len-1)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun processStr(s: String, k: Long): Char {
        var l = s.fold(0L) {r,c->when(c){'*'->max(0L,r-1);'%'->r;'#'->r+r;else->r+1}}
        var k = k
        if (k < l) for (c in s.reversed()) when (c) {
            '*' -> l++; '%' -> k = l - 1 - k; '#' -> { l /= 2; k %= l }
            else -> if (k == --l) return c
        }
        return '.'
    }
```
```rust
    pub fn process_str(s: String, mut k: i64) -> char {
        let mut l=s.chars().fold(0,|r,c|match c{'*'=>(r-1).max(0),'%'=>r,'#'=>r+r,_=>r+1});
        if k < l { for c in s.chars().rev() { match c {
            '*' => l += 1, '%' => k = l - 1 - k, '#' => { l /= 2; k %= l }
            _ => { l -= 1; if k == l { return c } }
        } } } '.'
    }
```

