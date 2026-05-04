---
layout: leetcode-entry
title: "796. Rotate String"
permalink: "/leetcode/problem/2026-05-03-796-rotate-string/"
leetcode_ui: true
entry_slug: "2026-05-03-796-rotate-string"
---

[796. Rotate String](https://leetcode.com/problems/rotate-string/solutions/8132732/kotlin-rust-by-samoylenkodmitry-gzdt/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03052026-796-rotate-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vmxn9z98EjM)

https://dmitrysamoylenko.com/leetcode/

![03.05.2026.webp](/assets/leetcode_daily_images/03.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1347

#### Problem TLDR

Match strings after rotation

#### Intuition

Brute-force: check every possible rotation
Clever idea: rotation is identical to repetition, find goal in concatenated source
KMP: precompute p[] array where p[i] is the `length` of matching goal[0..p[i]-1] == goal[i-(p[i]-1)..i]. Jump back j = p[j-1] if chars doesn't match.

#### Approach

* the `.contains` in Rust has O(n) time and O(1) space complexity and based on Two-Way String Matching algo (that is alien looking and based on math

#### Complexity

- Time complexity:
$$O(n^2|n)$$

- Space complexity:
$$O(1|n)$$

#### Code

```kotlin
    fun rotateString(s: String, g: String) =
    s.length==g.length && g in s+s
```
```rust
    pub fn rotate_string(s: String, g: String) -> bool {
        let (s,g) = (s.as_bytes(), g.as_bytes());
        let (mut p, mut j) = (vec![0; g.len()], 0);
        for i in 1..p.len() {
            while j > 0 && g[i] != g[j] { j = p[j-1] }
            if g[i] == g[j] { j += 1}; p[i] = j
        }; j = 0;
        p.len() == s.len() && (0..s.len() * 2).any(|i| {
            while j > 0 && s[i%s.len()] != g[j] { j = p[j-1] }
            if s[i%s.len()] == g[j] { j += 1 }; j == p.len()
        })
    }
```

