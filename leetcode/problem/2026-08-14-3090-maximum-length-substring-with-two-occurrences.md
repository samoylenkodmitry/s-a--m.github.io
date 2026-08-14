---
layout: leetcode-entry
title: "3090. Maximum Length Substring With Two Occurrences"
permalink: "/leetcode/problem/2026-08-14-3090-maximum-length-substring-with-two-occurrences/"
leetcode_ui: true
entry_slug: "2026-08-14-3090-maximum-length-substring-with-two-occurrences"
---

[3090. Maximum Length Substring With Two Occurrences](https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/solutions/8459944/kotlin-rust-by-samoylenkodmitry-0e0i/) easy
[substack](https://dmitriisamoilenko.substack.com/p/14082026-3090-maximum-length-substring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cD53oVbtqi8)

https://dmitrysamoylenko.com/leetcode/

![14.08.2026.webp](/assets/leetcode_daily_images/14.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1451

#### Problem TLDR

Max substring repeats less than 3

#### Intuition

Brute force. Try every length from largest to smalest.

#### Approach

* Rust: rfind
* Kotlin: find

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maximumLengthSubstring(s: String) = (s.length downTo 2)
    .find {s.windowed(it).any{w->w.all{w.count{c->c==it}<3}}}
```
```rust
    pub fn maximum_length_substring(s: String) -> i32 {
        (1..=s.len()).rfind(|&n|s.as_bytes().windows(n)
        .any(|w|w.iter().counts().values().all(|&v|v<3))).unwrap() as _
    }
```

