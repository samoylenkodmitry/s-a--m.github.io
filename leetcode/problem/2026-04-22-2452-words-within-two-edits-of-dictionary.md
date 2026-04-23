---
layout: leetcode-entry
title: "2452. Words Within Two Edits of Dictionary"
permalink: "/leetcode/problem/2026-04-22-2452-words-within-two-edits-of-dictionary/"
leetcode_ui: true
entry_slug: "2026-04-22-2452-words-within-two-edits-of-dictionary"
---

[2452. Words Within Two Edits of Dictionary](https://leetcode.com/problems/words-within-two-edits-of-dictionary/solutions/8045468/kotlin-rust-by-samoylenkodmitry-07or/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22042026-2452-words-within-two-edits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QG_Odt79-lU)

![22.04.2026.webp](/assets/leetcode_daily_images/22.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1336

#### Problem TLDR

Words in dictionary with 2 edits #medium

#### Intuition

1. We can place all single-edits of dictionary into a hash set, then check single-edits of words.
2. Or just brute-force with the same time complexity

#### Approach

* Rust has `retain`

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 28ms
    fun twoEditWords(q: Array<String>, d: Array<String>) =
    q.filter { q -> d.any { d -> d.indices.count { d[it] != q[it]} < 3 }}
```
```rust
// 1ms
    pub fn two_edit_words(mut q: Vec<String>, d: Vec<String>) -> Vec<String> {
        q.retain(|q| d.iter().any(|d| d.bytes().zip(q.bytes()).filter(|(d,q)|d!=q).count()<3)); q
    }
```

