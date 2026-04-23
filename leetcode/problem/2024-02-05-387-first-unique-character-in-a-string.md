---
layout: leetcode-entry
title: "387. First Unique Character in a String"
permalink: "/leetcode/problem/2024-02-05-387-first-unique-character-in-a-string/"
leetcode_ui: true
entry_slug: "2024-02-05-387-first-unique-character-in-a-string"
---

[387. First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/description) easy
[blog post](https://leetcode.com/problems/first-unique-character-in-a-string/solutions/4679671/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05022024-387-first-unique-character?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/q04HvyhZj8o)
![image.png](/assets/leetcode_daily_images/7fe7e9ef.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/496

#### Problem TLDR

First non-repeating char position.

#### Intuition

Compute char's frequencies, then find first of 1.

#### Approach

Let's try to make code shorter:
Kotlin:
* groupBy
* run
* indexOfFirst
Rust:
* vec![]
* String.find
* map_or

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun firstUniqChar(s: String) = s.groupBy { it }
    .run { s.indexOfFirst { this[it]!!.size < 2 } }

```
```rust

  pub fn first_uniq_char(s: String) -> i32 {
    let mut f = vec![0; 128];
    for b in s.bytes() { f[b as usize] += 1 }
    s.find(|c| f[c as usize] < 2).map_or(-1, |i| i as i32)
  }

```

