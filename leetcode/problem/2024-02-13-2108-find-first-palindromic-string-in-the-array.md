---
layout: leetcode-entry
title: "2108. Find First Palindromic String in the Array"
permalink: "/leetcode/problem/2024-02-13-2108-find-first-palindromic-string-in-the-array/"
leetcode_ui: true
entry_slug: "2024-02-13-2108-find-first-palindromic-string-in-the-array"
---

[2108. Find First Palindromic String in the Array](https://leetcode.com/problems/find-first-palindromic-string-in-the-array/description/) easy
[blog post](https://leetcode.com/problems/find-first-palindromic-string-in-the-array/solutions/4718669/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13022024-2108-find-first-palindromic?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/O6IEjBQftE8)
![image.png](/assets/leetcode_daily_images/37e43af7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/504

#### Problem TLDR

Find a palindrome.

#### Intuition

Compare first chars with the last.

#### Approach

Let's use some API's:
* Kotlin: `firstOrNull`, `all`
* Rust: `into_iter`, `find`, `chars`, `eq`, `rev`, `unwrap_or_else`, `into`. The `eq` compares two iterators with O(1) space.

#### Complexity

- Time complexity:
$$O(wn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun firstPalindrome(words: Array<String>) =
      words.firstOrNull { w ->
        (0..w.length / 2).all { w[it] == w[w.lastIndex - it] }
      } ?: ""

```
```rust

  pub fn first_palindrome(words: Vec<String>) -> String {
    words.into_iter().find(|w|
      w.chars().eq(w.chars().rev())
    ).unwrap_or_else(|| "".into())
  }

```

