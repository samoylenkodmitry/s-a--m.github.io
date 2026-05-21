---
layout: leetcode-entry
title: "3043. Find the Length of the Longest Common Prefix"
permalink: "/leetcode/problem/2026-05-21-3043-find-the-length-of-the-longest-common-prefix/"
leetcode_ui: true
entry_slug: "2026-05-21-3043-find-the-length-of-the-longest-common-prefix"
---

[3043. Find the Length of the Longest Common Prefix](https://leetcode.com/problems/find-the-length-of-the-longest-common-prefix/solutions/8283181/kotlin-rust-by-samoylenkodmitry-u706/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21052026-3043-find-the-length-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XgDXpQLkJXA)

https://dmitrysamoylenko.com/leetcode/

![21.05.2026.webp](/assets/leetcode_daily_images/21.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1366

#### Problem TLDR

Longest prefix of pairs

#### Intuition

* hashset: put all prefixes of one array, query all prefixes of the second array
* trie: put all prefixes of one array into trie, check longest path for all prefixes of the second

#### Approach

* we can use strings or ints

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun longestCommonPrefix(a: IntArray, b: IntArray) = run {
        val s = a.flatMap { "$it".scan("", String::plus) }.toSet()
        b.maxOf { var v = "$it"; while (v !in s) v = v.dropLast(1); v.length }
    }
```
```rust
    pub fn longest_common_prefix(a: Vec<i32>, b: Vec<i32>) -> i32 {
        #[derive(Default)] struct T(HashMap<u8, T>); let mut r = T::default();
        for x in a { x.to_string().bytes().fold(&mut r, |t, c| t.0.entry(c).or_default()); }
        b.iter().map(|x| {
            let mut t = &r;
            x.to_string().bytes().take_while(|c| t.0.get(c).map(|n| t = n).is_some()).count() as _
        }).max().unwrap()
    }
```

