---
layout: leetcode-entry
title: "1544. Make The String Great"
permalink: "/leetcode/problem/2024-04-05-1544-make-the-string-great/"
leetcode_ui: true
entry_slug: "2024-04-05-1544-make-the-string-great"
---

[1544. Make The String Great](https://leetcode.com/problems/make-the-string-great/description/) easy
[blog post](https://leetcode.com/problems/make-the-string-great/solutions/4976163/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05042024-1544-make-the-string-great?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/M3FXgXuF1CQ)

![2024-04-05_08-24.webp](/assets/leetcode_daily_images/fa839135.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/561

#### Problem TLDR

Remove lowercase-uppercase pairs #easy

#### Intuition
Consider example:
```j
    EbBe
     **
    E  e

```
After removing the middle `bB` we have to consider the remaining `Ee`. We can use Stack to do that.

#### Approach

In Kotlin: no need for Stack, just use StringBuilder.
In Rust: `Vec` can be used as a Stack. There is no `to_lowercase` method returning a char, however there is a `to_ascii_lowercase`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun makeGood(s: String) = buildString {
    for (c in s)
      if (length > 0 && c != get(lastIndex) &&
        c.lowercase() == get(lastIndex).lowercase()
      ) setLength(lastIndex) else append(c)
  }

```
```rust

  pub fn make_good(s: String) -> String {
    let mut stack = vec![];
    for c in s.chars() {
      if stack.is_empty() { stack.push(c) }
      else {
        let p = *stack.last().unwrap();
        if c != p && c.to_lowercase().eq(p.to_lowercase()) {
          stack.pop();
        } else { stack.push(c) }
      }
    }
    stack.iter().collect()
  }

```

