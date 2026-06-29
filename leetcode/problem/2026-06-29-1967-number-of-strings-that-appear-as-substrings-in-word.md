---
layout: leetcode-entry
title: "1967. Number of Strings That Appear as Substrings in Word"
permalink: "/leetcode/problem/2026-06-29-1967-number-of-strings-that-appear-as-substrings-in-word/"
leetcode_ui: true
entry_slug: "2026-06-29-1967-number-of-strings-that-appear-as-substrings-in-word"
---

[1967. Number of Strings That Appear as Substrings in Word](https://leetcode.com/problems/number-of-strings-that-appear-as-substrings-in-word/solutions/8364895/kotlin-rust-by-samoylenkodmitry-sobk/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29062026-1967-number-of-strings-that?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/b6MqVSIzuIQ)

https://dmitrysamoylenko.com/leetcode/

![29.06.2026.webp](/assets/leetcode_daily_images/29.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1405

#### Problem TLDR

N patterns in a word

#### Intuition

Brute-force.

#### Approach

* an optimal solutin exists, just ask your ai

#### Complexity

- Time complexity:
$$O(nw)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun numOfStrings(p: Array<String>, w: String) =
    p.count { it in w }
```
```rust
    pub fn num_of_strings(p: Vec<String>, w: String) -> i32 {
       p.iter().filter(|&p|w.contains(p)).count() as _
    }
```

