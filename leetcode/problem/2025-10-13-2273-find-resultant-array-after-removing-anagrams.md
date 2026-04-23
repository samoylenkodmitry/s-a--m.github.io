---
layout: leetcode-entry
title: "2273. Find Resultant Array After Removing Anagrams"
permalink: "/leetcode/problem/2025-10-13-2273-find-resultant-array-after-removing-anagrams/"
leetcode_ui: true
entry_slug: "2025-10-13-2273-find-resultant-array-after-removing-anagrams"
---

[2273. Find Resultant Array After Removing Anagrams](https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/description) medium
[blog post](https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/solutions/7271212/kotlin-rust-by-samoylenkodmitry-jtmb/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13102025-2273-find-resultant-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tAvFWsJgASQ)

![ef5404c8-652d-4e6c-ac76-4e95cfe41880 (1).webp](/assets/leetcode_daily_images/1cadd69b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1141

#### Problem TLDR

Dedup anagrams #easy

#### Intuition

Simulate the process.
The islands of equal-by-anagram are not influence each other when split by non-equal word.

#### Approach

* going from left to right, take value if previous is not anagram to current
* check anagrams by: a) sorting b) comparing the frequency map

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 42ms
    fun removeAnagrams(w: Array<String>) = w.take(1) + w.asList()
    .zipWithNext().mapNotNull {(a,b) -> b.takeIf{a.groupBy{it}!=b.groupBy{it}}}

```
```rust

// 3ms
    pub fn remove_anagrams(mut w: Vec<String>) -> Vec<String> {
       w.dedup_by_key(|w| w.bytes().counts()); w
    }

```

