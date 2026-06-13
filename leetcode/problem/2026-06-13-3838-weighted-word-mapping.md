---
layout: leetcode-entry
title: "3838. Weighted Word Mapping"
permalink: "/leetcode/problem/2026-06-13-3838-weighted-word-mapping/"
leetcode_ui: true
entry_slug: "2026-06-13-3838-weighted-word-mapping"
---

[3838. Weighted Word Mapping](https://leetcode.com/problems/weighted-word-mapping/solutions/8331152/kotlin-rust-by-samoylenkodmitry-j1m2/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13062026-3838-weighted-word-mapping?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/asQKj_XI-y8)

https://dmitrysamoylenko.com/leetcode/

![13.06.2026.webp](/assets/leetcode_daily_images/13.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1389

#### Problem TLDR

Convert words to chars by weighting their sums and reversing

#### Intuition

brute-force

#### Approach

* reverse by `z`-c

#### Complexity

- Time complexity:
$$O(nw)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun mapWordWeights(w: Array<String>, ws: IntArray) =
    w.joinToString(""){""+('z'-it.sumOf{ws[it-'a']}%26)}
```
```rust
    pub fn map_word_weights(w: Vec<String>, ws: Vec<i32>) -> String {
        w.iter().map(|s|(122-s.bytes().fold(0,|a,c|a+ws[(c-97)as usize])%26)as u8 as char).collect()
    }
```

