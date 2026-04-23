---
layout: leetcode-entry
title: "2000. Reverse Prefix of Word"
permalink: "/leetcode/problem/2024-05-01-2000-reverse-prefix-of-word/"
leetcode_ui: true
entry_slug: "2024-05-01-2000-reverse-prefix-of-word"
---

[2000. Reverse Prefix of Word](https://leetcode.com/problems/reverse-prefix-of-word/description/) easy
[blog post](https://leetcode.com/problems/reverse-prefix-of-word/solutions/5094699/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01052024-2000-reverse-prefix-of-word?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3HaIy6XeluA)
![2024-05-01_09-09.webp](/assets/leetcode_daily_images/728f271e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/589

#### Problem TLDR

Reverse `[..ch]` prefix in string #easy

#### Intuition

First find the position, then reverse the prefix.

#### Approach

Can you make the code shorter? (Don't do this in the interview, however, we skipped optimized case of not found index.)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun reversePrefix(word: String, ch: Char) = String(
        word.toCharArray().apply { reverse(0, indexOf(ch) + 1) }
    )

```
```rust

    pub fn reverse_prefix(mut word: String, ch: char) -> String {
        let i = word.find(ch).unwrap_or(0) + 1;
        word[..i].chars().rev().chain(word[i..].chars()).collect()
    }

```

