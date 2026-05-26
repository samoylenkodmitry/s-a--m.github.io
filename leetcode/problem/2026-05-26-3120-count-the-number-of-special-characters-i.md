---
layout: leetcode-entry
title: "3120. Count the Number of Special Characters I"
permalink: "/leetcode/problem/2026-05-26-3120-count-the-number-of-special-characters-i/"
leetcode_ui: true
entry_slug: "2026-05-26-3120-count-the-number-of-special-characters-i"
---

[3120. Count the Number of Special Characters I](https://leetcode.com/problems/count-the-number-of-special-characters-i/solutions/8294319/kotlin-rust-by-samoylenkodmitry-ogff/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26052026-3120-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1qIg9iPsFYo)

https://dmitrysamoylenko.com/leetcode/

![26.05.2026.webp](/assets/leetcode_daily_images/26.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1371

#### Problem TLDR

Count letters with both cases

#### Intuition

Brute-force is accepted.
O(n) memory: for any uniq lowercase check if uppercase present, use hashset
O(1) memory: two bitmasks, for lower and for upper cases; (m & M) count bits is the result

#### Approach

* regex is ugly here

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun numberOfSpecialChars(w: String) =
        w.toSet().count {it-32 in w}
```
```rust
    pub fn number_of_special_chars(w: String) -> i32 {
        ('a'..='z').zip('A'..='Z').filter(|&(c,C)| w.contains(c)&&w.contains(C)).count() as _
    }
```

