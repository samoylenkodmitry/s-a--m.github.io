---
layout: leetcode-entry
title: "2942. Find Words Containing Character"
permalink: "/leetcode/problem/2025-05-24-2942-find-words-containing-character/"
leetcode_ui: true
entry_slug: "2025-05-24-2942-find-words-containing-character"
---

[2942. Find Words Containing Character](https://leetcode.com/problems/find-words-containing-character/description/) easy
[blog post](https://leetcode.com/problems/find-words-containing-character/solutions/6775555/kotlin-rust-by-samoylenkodmitry-p4db/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24052025-2942-find-words-containing?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9DTaFrwQoxE)
![1.webp](/assets/leetcode_daily_images/c4c7a869.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/998

#### Problem TLDR

Indices with x #easy

#### Intuition

Do what is asked

#### Approach

* the answer can be in `any order` suggests some interesting optimizations: what if we unroll loops or even start work in parallel? (however, in Kotlin I wasnt able to gain any performance)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 25ms
    fun findWordsContaining(w: Array<String>, x: Char) =
        w.indices.filter { x in w[it] }

```
```kotlin

// 9ms
    fun findWordsContaining(w: Array<String>, x: Char): List<Int> {
        val res = ArrayList<Int>(w.size)
        for (i in w.indices) if (x in w[i]) res += i
        return res
    }

```
```kotlin

// 3ms
    fun findWordsContaining(w: Array<String>, x: Char): List<Int> {
        val res = ArrayList<Int>(w.size)
        for (i in w.indices)
            for (c in w[i]) if (c == x) { res += i; break }
        return res
    }

```
```rust

// 0ms
    pub fn find_words_containing(w: Vec<String>, x: char) -> Vec<i32> {
        (0..w.len()).filter(|&i| w[i].contains(x)).map(|i| i as _).collect()
    }

```
```c++

// 0ms
    vector<int> findWordsContaining(vector<string>& w, char x) {
        vector<int> r;
        for (int i = 0; i < size(w); ++i) if (w[i].contains(x)) r.push_back(i);
        return r;
    }

```

