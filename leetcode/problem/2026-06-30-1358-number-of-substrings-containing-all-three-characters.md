---
layout: leetcode-entry
title: "1358. Number of Substrings Containing All Three Characters"
permalink: "/leetcode/problem/2026-06-30-1358-number-of-substrings-containing-all-three-characters/"
leetcode_ui: true
entry_slug: "2026-06-30-1358-number-of-substrings-containing-all-three-characters"
---

[1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/solutions/8367163/kotlin-rust-by-samoylenkodmitry-p0vp/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30062026-1358-number-of-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-TTBLi0EltY)

https://dmitrysamoylenko.com/leetcode/

![30.06.2026.webp](/assets/leetcode_daily_images/30.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1406

#### Problem TLDR

Substrings with 3 letters

#### Intuition

Sum of substrings ending at position i. Count all prefixes.

#### Approach

* just remember the latest position of each letter, the min would be the prefix

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun numberOfSubstrings(s: String) = IntArray(3).run {
        s.indices.sumOf { i -> set(s[i]-'a', i+1); min() }
    }
```
```rust
    pub fn number_of_substrings(s: String) -> i32 {
        s.bytes().zip(1..).fold(([0;3],0),|(mut j,mut s),(b,i)|{
            j[b as usize%3]=i;(j,s+j[0].min(j[1]).min(j[2]))}).1
    }
```

