---
layout: leetcode-entry
title: "3612. Process String with Special Operations I"
permalink: "/leetcode/problem/2026-06-16-3612-process-string-with-special-operations-i/"
leetcode_ui: true
entry_slug: "2026-06-16-3612-process-string-with-special-operations-i"
---

[3612. Process String with Special Operations I](https://leetcode.com/problems/process-string-with-special-operations-i/solutions/8337391/kotlin-rust-by-samoylenkodmitry-piwo/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16062026-3612-process-string-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FRj0YKp_f7c)

https://dmitrysamoylenko.com/leetcode/

![16.06.2026.webp](/assets/leetcode_daily_images/16.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1392

#### Problem TLDR

Pattern-build a string, %-reverse, #-repeat, *-pop

#### Intuition

Just simulate the rules. In a worst case we would have 2^n time/space complexity if every letter would be c#..#

#### Approach

* we can use fold
* Rust has `extend_from_within(0..)`

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun processStr(s: String) = s.fold("") { r, c ->
        when (c) {
            '*' -> r.dropLast(1); '%' -> r.reversed()
            '#' -> r + r; else -> r + c
        }
    }
```
```rust
    pub fn process_str(s: String) -> String {
        let mut res = vec![];
        for b in s.bytes() { match b {
            b'*' => {res.pop();}, b'%' => res.reverse(),
            b'#' => res.extend_from_within(0..), _ => res.push(b)
        }}
        String::from_utf8(res).unwrap()
    }
```

