---
layout: leetcode-entry
title: "3121. Count the Number of Special Characters II"
permalink: "/leetcode/problem/2026-05-27-3121-count-the-number-of-special-characters-ii/"
leetcode_ui: true
entry_slug: "2026-05-27-3121-count-the-number-of-special-characters-ii"
---

[3121. Count the Number of Special Characters II](https://leetcode.com/problems/count-the-number-of-special-characters-ii/solutions/8296584/kotlin-rust-by-samoylenkodmitry-0oo3/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27052026-3121-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ngzq2yMD4ZY)

https://dmitrysamoylenko.com/leetcode/

![27.05.2026.webp](/assets/leetcode_daily_images/27.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1372

#### Problem TLDR

Count sorted characters

#### Intuition

Brute-force: checkk all letters separately a..z, last index of c should be in range of 0..first index of C
Optimal: bitmasks to track visited uppercase and lowercase and invalid marker

#### Approach

* 0.. is essential in kotlin, because lastIndexof can be -1

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun numberOfSpecialChars(w: String) = ('a'..'z')
    .count { w.lastIndexOf(it) in 0..<w.indexOf(it-32) }
```
```rust
    pub fn number_of_special_chars(w: String) -> i32 {
        (b'a'..=b'z').filter(|&c| w.rfind(c as char)
        .is_some_and(|l| Some(l) < w.find((c - 32) as char))).count() as _
    }
```

