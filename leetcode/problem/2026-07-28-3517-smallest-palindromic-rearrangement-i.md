---
layout: leetcode-entry
title: "3517. Smallest Palindromic Rearrangement I"
permalink: "/leetcode/problem/2026-07-28-3517-smallest-palindromic-rearrangement-i/"
leetcode_ui: true
entry_slug: "2026-07-28-3517-smallest-palindromic-rearrangement-i"
---

[3517. Smallest Palindromic Rearrangement I](https://leetcode.com/problems/smallest-palindromic-rearrangement-i/solutions/8425649/kotlin-rust-by-samoylenkodmitry-ur04/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28072026-3517-smallest-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VDUrEFWlksE)

https://dmitrysamoylenko.com/leetcode/

![28.07.2026.webp](/assets/leetcode_daily_images/28.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1434

#### Problem TLDR

Smallest palindrome rearrangement

#### Intuition

Take first half and sort it.

#### Approach

* tip: prepare yourself for the k-th rearrangement

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun smallestPalindrome(s: String) = run {
        val h = s.take(s.length/2).map{it}.sorted().joinToString("")
        h + (if (s.length%2>0) s[s.length/2] else "") + h.reversed()
    }
```
```rust
    pub fn smallest_palindrome(s: String) -> String {
        let n = s.len(); let h = s[..n/2].chars().sorted();
        h.clone().chain(s[n/2..(n+1)/2].chars()).chain(h.rev()).collect()
    }
```

