---
layout: leetcode-entry
title: "2785. Sort Vowels in a String"
permalink: "/leetcode/problem/2025-09-11-2785-sort-vowels-in-a-string/"
leetcode_ui: true
entry_slug: "2025-09-11-2785-sort-vowels-in-a-string"
---

[2785. Sort Vowels in a String](https://leetcode.com/problems/sort-vowels-in-a-string/description) medium
[blog post](https://leetcode.com/problems/sort-vowels-in-a-string/solutions/7178103/kotlin-rust-by-samoylenkodmitry-zl36/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11092025-2785-sort-vowels-in-a-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1zQ0Ee-Lk8w)

![1.webp](/assets/leetcode_daily_images/99ffd029.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1109

#### Problem TLDR

Sort vowels #medium

#### Intuition

Just implementation, no extra tricks.

#### Approach

* copy vowels, sort, put back
* or do a counting sort

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 102ms
    fun sortVowels(s: String) = buildString {
        val vw = s.filter { it in "aeiouAEIOU" }.toList().sorted()
        var i = 0
        for (c in s) append(if (c in "aeiouAEIOU") vw[i++] else c)
    }

```
```kotlin

// 27ms
    fun sortVowels(s: String) = buildString {
        val v = "AEIOUaeiou"; val vw = IntArray(12)
        for (c in s) ++vw[1 + v.indexOf(c)]
        for (c in s) append(if (c in v)
            v[(0..10).first {vw[it+1] > 0}.also {--vw[it+1]}] else c)
    }

```
```rust

// 10ms
    pub fn sort_vowels(s: String) -> String {
        let mut t = s.chars().filter(|&c| "AEIOUaeiou".contains(c)).sorted();
        s.chars().map(|c| if "AEIOUaeiou".contains(c) { t.next().unwrap() } else { c }).collect()
    }

```

