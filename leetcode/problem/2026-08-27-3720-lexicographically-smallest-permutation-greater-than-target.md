---
layout: leetcode-entry
title: "3720. Lexicographically Smallest Permutation Greater Than Target"
permalink: "/leetcode/problem/2026-08-27-3720-lexicographically-smallest-permutation-greater-than-target/"
leetcode_ui: true
entry_slug: "2026-08-27-3720-lexicographically-smallest-permutation-greater-than-target"
---

[3720. Lexicographically Smallest Permutation Greater Than Target](https://leetcode.com/problems/lexicographically-smallest-permutation-greater-than-target/solutions/8485663/kotlin-rust-by-samoylenkodmitry-tyq1/) medium
[substack](https://dmitriisamoilenko.substack.com/p/27082026-3720-lexicographically-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/0kfL56tMvjc)

https://dmitrysamoylenko.com/leetcode/

![27.08.2026.webp](/assets/leetcode_daily_images/27.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1464

#### Problem TLDR

Smallest permutation bigger than target

#### Intuition

Try to one-up every position. The tail is just sorted remainder.

#### Approach

* to match the prefix adjust the frequency; exit at the first mismatch

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun lexGreaterPermutation(s: String, t: String): String {
        val f = IntArray(128); for (c in s) ++f[c.code]; var r = ""
        for (i in t.indices) {
            (t[i] + 1..'z').find { f[it.code] > 0 }?.run {
                f[code]--
                r = t.take(i) + this + ('a'..'z').joinToString("") { "$it".repeat(f[it.code]) }
                f[code]++
            }
            if (f[t[i].code]-- == 0) break
        };   return r
    }
```
```rust
    pub fn lex_greater_permutation(s: String, t: String) -> String {
        let (mut f, mut r) = ([0; 123], "".into()); for b in s.bytes() { f[b as usize] += 1 }
        for (i, b) in t.bytes().enumerate() {
            if let Some(c) = (b + 1..=122).find(|&c| f[c as usize] > 0) {
                r = format!("{}{}{}", &t[..i], c as char, (97..=122).flat_map(|x| vec![x as char; f[x as usize] - (x == c) as usize]).join(""));
            }
            if f[b as usize] == 0 { break }; f[b as usize] -= 1;
        } r
    }
```

