---
layout: leetcode-entry
title: "2904. Shortest and Lexicographically Smallest Beautiful String"
permalink: "/leetcode/problem/2026-08-26-2904-shortest-and-lexicographically-smallest-beautiful-string/"
leetcode_ui: true
entry_slug: "2026-08-26-2904-shortest-and-lexicographically-smallest-beautiful-string"
---

[2904. Shortest and Lexicographically Smallest Beautiful String](https://leetcode.com/problems/shortest-and-lexicographically-smallest-beautiful-string/solutions/8483667/kotlin-rust-by-samoylenkodmitry-7i4a/) medium
[substack](https://dmitriisamoilenko.substack.com/p/26082026-2904-shortest-and-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/51uBUxFc5yE)

https://dmitrysamoylenko.com/leetcode/

![26.08.2026.webp](/assets/leetcode_daily_images/26.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1463

#### Problem TLDR

Smallest substring with k-ones

#### Intuition

Two pointers: always move the right, move the left while it is safe to shrink.
Problem size is small, brute force is accepted.

#### Approach

* first take smallest length, then compare lexicographically

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun shortestBeautifulSubstring(s: String, k: Int) =
    (k..s.length).firstNotNullOfOrNull { len ->
        s.windowed(len).filter { w -> w.count {it>'0'} == k }.minOrNull()
    } ?: ""
```
```rust
    pub fn shortest_beautiful_substring(s: String, k: i32) -> String {
        (k as usize..=s.len()).find_map(|l| (0..=s.len() - l)
            .map(|i| &s[i..i + l])
            .filter(|w| w.matches('1').count() == k as usize)
            .min()
        ).unwrap_or("").into()
    }
```

